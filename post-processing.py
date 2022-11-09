from __future__ import print_function
import sys
import os

import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import argparse
from fisher import pvalue
from sklearn.cluster import KMeans

from desmond_io import prepare_input_data, read_bic_table
from method_post import merge_biclusters


def get_rand_subnet(network, g_size, max_n_restarts=10):
    n = random.choice(network.nodes())
    genes = set([n])
    n_restatrs = 1
    while len(genes) < g_size:
        if n_restatrs > max_n_restarts:
            print("Cannot generate random subnetwork.Try decreasing g_size" +
                  " or increasing max_n_restarts", file=sys.stderr)
            return None
        candidates = [x for x in network[n].keys() if x not in genes]
        if len(candidates) > 0:
            n = random.choice(candidates)
            genes.add(n)
        else:
            candidates2 = []
            for g in genes:
                neighbors = [x for x in network[g].keys() if x not in genes]
                if len(neighbors) > 1:
                    candidates2.append(g)
            if len(candidates2) > 0:
                n = random.choice(candidates2)
            else:
                n = random.choice(network.nodes())
                genes = set([n])
                n_restatrs += 1
    return list(genes)


def generate_null_dist(exprs, network, g_size=3, n_permutations=100):
    """Generates N random subnetworks of given size"""
    # no more than 1000 permuations
    null_dist_SNR = []
    null_dist_psize = []
    for p in range(0, n_permutations):
        # if p%1000 == 0:
        #    print("\t\tg_size:", g_size, "iteraton:",p)
        e = exprs[get_rand_subnet(network, g_size), :]
        labels = KMeans(n_clusters=2, random_state=0, n_init=1,
                        max_iter=100).fit(e.T).labels_
        ndx0 = np.where(labels == 0)[0]
        ndx1 = np.where(labels == 1)[0]
        if len(ndx1) < len(ndx0):
            p_size = len(ndx1)
        else:
            p_size = len(ndx0)
        e1 = e[:, ndx1]
        e0 = e[:, ndx0]
        SNR = np.mean(abs(e0.mean(axis=1)-e1.mean(axis=1)) /
                      (e0.std(axis=1)+e1.std(axis=1)))
        null_dist_psize.append(p_size)
        null_dist_SNR.append(SNR)
    return null_dist_SNR, null_dist_psize


def calc_not_lesser_bics(SNR, p_size, null_dist_SNR, null_dist_psize):
    # caluclates the number of non-smaller biclusters
    # with more samples and higher SNR
    n_not_lesser = 0
    n_higher_SNR = 0
    N = len(null_dist_psize)
    for i in range(0, N):
        if null_dist_psize[i] >= p_size:
            n_not_lesser += 1
            if null_dist_SNR[i] >= SNR:
                n_higher_SNR += 1
    return pd.Series({"n_not_less_samples": n_not_lesser,
                      "n_higher_SNR": n_higher_SNR,
                      "p_val": (n_higher_SNR+1)*1.0/(N+1)})


def calc_gene_overlap_pval(bic, bic2, N):
    g1 = bic["genes"]
    g2 = bic2["genes"]
    g1_g2 = len(g1.intersection(g2))
    g1_only = len(g1.difference(g2))
    g2_only = len(g2.difference(g1))
    p_val = pvalue(g1_g2, g1_only, g2_only, N-g1_g2-g1_only-g2_only).right_tail
    return p_val


def summarize_DESMOND_results(bic_up_file, bic_down_file, exprs, exprs_data,
                              g_names2ints, s_names2ints,
                              min_SNR=0.5, min_n_samples=5, verbose=True,
                              merge_discordant=False):
    '''1) merge biclusters containg exactly the same genes \n
    2) revert bicluster if it contains more than 1/2 samples'''

    if os.path.exists(bic_up_file):
        bic_up_df = read_bic_table(bic_up_file)
        bic_up_df["genes"] = bic_up_df["genes"].apply(
            lambda x: set(map(lambda y: g_names2ints[y], x)))
        bic_up_df["samples"] = bic_up_df["samples"].apply(
            lambda x: set(map(lambda y: s_names2ints[y], x)))
        bic_up = bic_up_df.T.to_dict()
    else:
        bic_up = {}

    if os.path.exists(bic_down_file):
        bic_down_df = read_bic_table(bic_down_file)
        bic_down_df["genes"] = bic_down_df["genes"].apply(
            lambda x: set(map(lambda y: g_names2ints[y], x)))
        bic_down_df["samples"] = bic_down_df["samples"].apply(
            lambda x: set(map(lambda y: s_names2ints[y], x)))

        bic_down = bic_down_df.T.to_dict()
    else:
        bic_down = {}

    if verbose:
        print("\tUP:", len(bic_up.keys()))
        print("\tDOWN:", len(bic_down.keys()))
    if bic_up.keys() == 0:
        return bic_down_df
    if bic_down.keys() == 0:
        return bic_up_df

    all_samples = set(range(exprs.shape[1]))

    # rename keys in down-regulated bics
    for k in bic_down.keys():
        bic_down[k]["id"] = bic_down[k]["id"] + len(bic_up.keys())

    # try merging overlapping biclusters
    merged_bics = merge_biclusters(
        list(bic_up.values())+list(bic_down.values()), exprs, exprs_data,
        min_n_samples=min_n_samples,
        min_SNR=min_SNR,
        pval_threshold=0.05,
        verbose=verbose,
        merge_discordant=merge_discordant)

    df = pd.DataFrame(merged_bics)
    df = df.sort_values("avgSNR", ascending=False)
    # new indexing
    df.index = range(0, df.shape[0])
    df_ = df.loc[df["n_samples"] > N/2, :]
    # new sample sets
    df.loc[df_.index, "samples"] = df.loc[df_.index, "samples"].apply(
        lambda x: all_samples.difference(x))
    df.loc[df_.index, "n_samples"] = df.loc[df_.index, "samples"].apply(len)
    # change direction
    up2down = df_.loc[df_["direction"] == "UP", :].index.values
    df.loc[up2down, "direction"] = "DOWN"
    down2up = df_.loc[df_["direction"] == "DOWN", :].index.values
    df.loc[down2up, "direction"] = "UP"
    if verbose:
        print("Total biclusters:", df.shape[0])
    return df


def relabel_exprs_and_network(exprs, network):
    'Changes gene and sample names to ints'
    g_names2ints = dict(zip(exprs.index.values, range(0, exprs.shape[0])))
    ints2g_names = exprs.index.values
    s_names2ints = dict(zip(exprs.columns.values, range(0, exprs.shape[1])))
    ints2s_names = exprs.columns.values
    exprs.rename(index=g_names2ints, columns=s_names2ints, inplace=True)
    network = nx.relabel_nodes(network, g_names2ints)
    return exprs, network, ints2g_names, ints2s_names, g_names2ints, s_names2ints


parser = argparse.ArgumentParser(
    description="Generates random subnetworks of size given sizes and " +
    "calculates empirical p-values for obsered avg. |SNR| values.",
    formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-e', '--exprs', dest='exprs_file', type=str,
                    help='Expression matrix with sample name in columns and ' +
                    'gene in rows, tab-separated.', default='',
                    required=True, metavar='exprs_zscores.tsv')
parser.add_argument('-n', '--network', dest='network_file', type=str,
                    help='Network in tab or NDEx2 format.', default='',
                    required=True, metavar='network.tab')
parser.add_argument('-u', '--up', dest='up', type=str,
                    help='Up-regulated biclusters.', default='',
                    required=False)
parser.add_argument('-d', '--down', dest='down', type=str,
                    help='Down-regulated biclusters.', default='',
                    required=False)
parser.add_argument('-o', '--out', dest='out', type=str,
                    help='Output file name.', default='.', required=False)
parser.add_argument('-s', '--SNR_file', dest='snr_file', type=str,
                    help='File with min_SNR treshold recorded.', required=True)
parser.add_argument('-ns', '--min_n_samples', dest='min_n_samples', type=int,
                    help='Minimal number of samples on edge. If not ' +
                    'specified, set to max(10,0.1*cohort_size).',
                    default=0, required=False)
parser.add_argument('--merge_discordant', dest='merge_discordant',
                    action='store_true',
                    help='Whether to merge up- and down-regulated ' +
                    'biclustert if their sample sets are similar.',
                    required=False)


parser.add_argument('-rn', dest='n_permutations', type=int,
                    help='Number of random subnetworks sampled.', default=1000,
                    required=False)

parser.add_argument('--verbose', dest='verbose',
                    action='store_true', help='', required=False)

start_time = time.time()

args = parser.parse_args()

exprs, network = prepare_input_data(
    args.exprs_file, args.network_file, verbose=args.verbose)
# change gene and sample names to ints
exprs, network, ints2g_names, ints2s_names, g_names2ints, s_names2ints = relabel_exprs_and_network(
    exprs, network)

# prepare expression files
exprs = exprs.values
exprs_sums = exprs.sum(axis=1)
exprs_sq_sums = np.square(exprs).sum(axis=1)
N = exprs.shape[1]
exprs_data = N, exprs_sums, exprs_sq_sums


if args.min_n_samples == 0:
    args.min_n_samples = max(10, int(0.1*exprs.shape[1]))
    if args. verbose:
        print("Mininal number of samples in a module:",
              args.min_n_samples, file=sys.stdout)

f = open(args.snr_file, "r")
min_SNR = float(f.readlines()[0].rstrip())
print("avg.SNR threshold:", min_SNR)

t0 = time.time()
print("time:\t Inputs read in %s s." %
      (round(time.time()-start_time, 2)), file=sys.stdout)
results = summarize_DESMOND_results(args.up, args.down, exprs, exprs_data,
                                    g_names2ints, s_names2ints,
                                    min_SNR=min_SNR,
                                    min_n_samples=args.min_n_samples,
                                    merge_discordant=args.merge_discordant,
                                    verbose=args.verbose)
print("time:\t Up- and down-regulated biclusets united in %s s." %
      (round(time.time()-t0, 2)), file=sys.stdout)

t0 = time.time()
results_p = []
g_sizes = results.groupby(by=["n_genes"])["n_genes"].count()
for g_size in g_sizes.index:
    t0 = time.time()
    # n_random_modules = g_sizes[g_size]*args.n_permutations
    if args.verbose:
        print("\t\tn_genes", g_size, "create", args.n_permutations,
              "random subnetworks...", file=sys.stdout)
    null_dist_SNR, null_dist_psize = generate_null_dist(
        exprs, network,
        g_size=g_size,
        n_permutations=args.n_permutations)

    df = results.loc[results["n_genes"] == g_size, :]
    df2 = df.apply(lambda row: calc_not_lesser_bics(
        row["avgSNR"], row["n_samples"], null_dist_SNR, null_dist_psize),
        axis=1)
    results_p.append(pd.concat([df, df2], axis=1))
    if args.verbose:
        print("\truntime", round(time.time()-t0, 2), "s", file=sys.stdout)

print("time:\t Random subnetworks " +
      "(%s per each bicluster size) created in %s s." % (
          args.n_permutations,
          round(time.time()-t0, 2)),
      file=sys.stdout)
results_p = pd.concat(results_p, axis=0)

results_p["genes"] = results_p["genes"].apply(
    lambda x: " ".join(sorted(map(lambda y: ints2g_names[y], x))))
results_p["samples"] = results_p["samples"].apply(
    lambda x: " ".join(sorted(map(lambda y: ints2s_names[y], x))))
results_p = results_p[["id", "n_genes", "n_samples", "avgSNR", "direction",
                       "p_val", "n_not_less_samples", "n_higher_SNR",
                       "genes", "samples"]]
results_p.loc[:, "id"] = range(0, results_p.shape[0])
results_p.set_index("id", drop=True, inplace=True)
results_p.sort_values(by="n_genes", ascending=False).sort_values(
    by="n_genes", ascending=False)
results_p.to_csv(args.out, sep="\t")
print("Total runtime:\t %s s." %
      (round(time.time()-start_time, 2)), file=sys.stdout)
