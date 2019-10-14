from __future__ import print_function
import sys,os

import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import argparse
from sklearn.cluster import KMeans

#sys.path.append(os.path.abspath("/home/olya/SFU/Breast_cancer/DESMOND/"))

from desmond_io import prepare_input_data
from evaluation import summarize_DESMOND_results

def get_rand_subnet(network, g_size, max_n_restarts=10):
    n = random.choice(network.nodes())
    genes = set([n])
    n_restatrs = 1
    while len(genes)<g_size:
        if n_restatrs > max_n_restarts:
            print("Cannot generate random subnetwork.Try decreasing g_size or increasing max_n_restarts", file=sys.stderr)
            return None
        candidates = [x for x in network[n].keys() if x not in genes]
        if len(candidates)>0:
            n = random.choice(candidates)
            genes.add(n)
        else:
            candidates2 = []
            for g in genes:
                neighbors = [x for x in network[g].keys() if x not in genes]
                if len(neighbors)>1:
                    candidates2.append(g)
            if len(candidates2)>0:
                n = random.choice(candidates2)
            else:
                n = random.choice(network.nodes())
                genes = set([n])
                n_restatrs += 1
    return genes

def generate_null_dist(exprs,network,g_size=3,n_permutations=100):
    """Generates N random subnetworks of given size"""
    # no more than 1000 permuations
    null_dist_SNR = []
    null_dist_psize = []
    for p in range(0,n_permutations):
        #if p%1000 == 0:
        #    print("\t\tg_size:", g_size, "iteraton:",p)
        e = exprs.loc[get_rand_subnet(network, g_size),:]
        labels = KMeans(n_clusters=2, random_state=0,n_init=1,max_iter=100).fit(e.T).labels_
        ndx0 = np.where(labels == 0)[0]
        ndx1 = np.where(labels == 1)[0]
        if len(ndx1) < len(ndx0):
            p_size = len(ndx1)
        else:
            p_size = len(ndx0)
        e1 = e.iloc[:,ndx1]
        e0 = e.iloc[:,ndx0]
        SNR = np.mean(abs(e0.mean(axis=1)-e1.mean(axis=1))/(e0.std(axis=1)+e1.std(axis=1)))
        null_dist_psize.append(p_size)
        null_dist_SNR.append(SNR)
    return null_dist_SNR, null_dist_psize

def calc_empirical_pval(SNR, p_size, null_dist_SNR, null_dist_psize):
    # caluclates the number of more extereme modules in null distribution 
    # with more samples and higher SNR
    n_more_extremal = 0
    for i in range(0, len(null_dist_psize)):
        if null_dist_SNR[i] >= SNR and null_dist_psize[i] >= p_size:
            n_more_extremal +=1
    return n_more_extremal*1.0/len(null_dist_SNR) 


parser = argparse.ArgumentParser(description="""Generates random subnetworks of size given sizes and calculates empirical p-values for obsered avg. |SNR| values.""" , formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-e','--exprs', dest='exprs_file', type=str, help='Expression matrix with sample name in columns and gene in rows, tab-separated.', default='', required=True,metavar='exprs_zscores.tsv')
parser.add_argument('-n','--network', dest='network_file', type=str, help='Network in tab or NDEx2 format.', default='', required=True, metavar='network.tab')
parser.add_argument('-u','--up', dest='up', type=str, help='Up-regulated biclusters.', default='', required=False)
parser.add_argument('-d','--down', dest='down', type=str, help='Down-regulated biclusters.', default='', required=False)
parser.add_argument('-o','--out', dest='out', type=str, help='Output file name.', default='.', required=False)

parser.add_argument('-rn', dest='n_permutations', type=int, help='Number of random subnetworks sampled.', default=100, required=False)

parser.add_argument('--verbose', dest='verbose', action='store_true', help='', required=False)

start_time = time.time()

args = parser.parse_args()

exprs, network = prepare_input_data(args.exprs_file,args.network_file, verbose = args.verbose)

results = summarize_DESMOND_results(args.up, args.down, args.exprs_file, verbose = args.verbose)

results_p = []
g_sizes = results.groupby(by=["n_genes"])["n_genes"].count()
for g_size in g_sizes.index:
    t0 = time.time()
    n_random_modules = g_sizes[g_size]*args.n_permutations
    if args.verbose:
        print("n_genes",g_size, "create",n_random_modules,"random subnetworks...", file= sys.stdout)
    null_dist_SNR, null_dist_psize = generate_null_dist(exprs,network,g_size=g_size,
                                   n_permutations=n_random_modules)
    
    df = results.loc[results["n_genes"]==g_size,:]
    df.loc[:,"p_val"] = df.apply(lambda row: calc_empirical_pval(row["avgSNR"], row["n_samples"],null_dist_SNR, null_dist_psize),axis =1)
    results_p.append(df)
    if args.verbose:
        print("\truntime", round(time.time()-t0,2),"s", file= sys.stdout)
results_p = pd.concat(results_p, axis =0)

results_p["genes"] = results_p["genes"].apply(lambda x:" ".join(map(str,x)))
results_p["samples"] = results_p["samples"].apply(lambda x:" ".join(map(str,x)))
results_p.to_csv(args.out, sep = "\t")