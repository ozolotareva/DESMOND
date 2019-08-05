from __future__ import print_function
import sys,os
import pandas as pd
from scipy.stats import fisher_exact, hypergeom
from fisher import pvalue
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import itertools
import warnings
import time
import seaborn as sns
import copy
import random
import gseapy as gp
from gseapy.stats import calc_pvalues, multiple_testing_correction

sys.path.append(os.path.abspath("/home/olya/SFU/Breast_cancer/breast_cancer_subtypes/"))
from method import bicluster_avg_SNR

#####  read resulted biclusters #######

def read_bicge(bicge_file, exprs):
    bics = []
    with open(bicge_file,"r") as infile:
        for line in infile.readlines():
            line = line.rstrip().split()
            if len(line)>0:
                if line[0] == 'Gene_names:':
                    bic = {}
                    genes = line[1:]
                    bic["genes"] = set(genes)
                if line[0] == 'Condition_names:':
                    pats = line[1:]
                    bic["patients"] = set(pats)
                    bic["avgSNR"] = bicluster_avg_SNR(pats, genes, exprs, absolute = True )
                    bics.append(bic)
    return bics


def read_B2PS(b2ps_p_file,b2ps_t_file, exprs):
    bics = []
    pat_labels = pd.read_csv(b2ps_p_file, header=None)
    pat_labels.index = exprs.columns
    gene_labels = pd.read_csv(b2ps_t_file, header=None)#[0].values
    gene_labels.index = exprs.index
    pat_dict = {}
    for s in set(pat_labels[0].values):
        pat_dict[s] = set(pat_labels.loc[pat_labels[0]==s,:].index.values)
    gene_dict = {}
    for s in set(gene_labels[0].values):
        gene_dict[s] = set(gene_labels.loc[gene_labels[0]==s,:].index.values)
    print("gene clusters:", len(gene_dict.keys()), "patient clusters:", len(pat_dict.keys()), file=sys.stderr)

    for g_clust in gene_dict.keys():
        best_pat_set, best_SNR = find_opt_SNR_threshold(exprs,gene_dict,pat_dict,g_clust)
        bics.append({"genes":set(gene_dict[g_clust]),
                     "patients": best_pat_set,"avgSNR":best_SNR })
    return bics

def find_opt_SNR_threshold(exprs,gene_dict,pat_dict,g_clust):
    all_pats = set(exprs.columns.values)
    genes = gene_dict[g_clust]
    #print("\t".join(genes))
    per_clust_mean_z = []
    for p_clust in pat_dict.keys():
        pats = pat_dict[p_clust]
        mean_z = np.mean(exprs.loc[genes, pats].mean())
        per_clust_mean_z.append(mean_z)
    order = [x for y,x in sorted(zip(per_clust_mean_z,pat_dict.keys()))]
    best_SNR = 0
    best_pat_set = set()
    for i in range(1, len(order)):
        pats = set()
        for p_clust in order[:i]:
            pats = pats | pat_dict[p_clust]
        avgSNR = bicluster_avg_SNR(pats, genes, exprs, absolute = True)
        if avgSNR > best_SNR:
            best_SNR = avgSNR
            if len(pats) > len(all_pats)/2:
                best_pat_set =  all_pats.difference(pats)
            else:
                best_pat_set = set(pats)
    return best_pat_set, best_SNR

########## Plotting bicluster statistics  ##############
def plot_n_bics(methods_dict,methods = [],datasets = ["TCGA-RNAseq","TCGA-micro", "METABRIC"],colors = ['blue','lightblue','lightgreen'],barWidth = 0.2):
    n_bics_produced = {}
    for ds in datasets:
        n_bics_produced[ds] = []
        for method in methods :
            n_bics = len(methods_dict[method][ds])
            n_bics_produced[ds].append(n_bics )
    # The x position of bars
    # first position
    bar_positions = [np.arange(len(n_bics_produced[ds]))]
    # from 1 to n
    for i in range(1, len(datasets)):
        bar_positions.append( [x + barWidth for x in bar_positions[i-1]])

    # Create  bars
    for i in range(len(datasets)):
        plt.bar(bar_positions[i],  n_bics_produced[datasets[i]], width = barWidth, color = colors[i],
                edgecolor = 'grey', capsize=7, label=datasets[i])

    # general layout
    plt.xticks([r + barWidth for r in range(len(n_bics_produced[datasets[0]]))], methods)
    plt.title('number of modules produced')
    plt.ylabel('modules')
    plt.legend()

def plot_distributions(methods_dict, methods = [],what="samples", how="size",
                            datasets = ["TCGA-RNAseq","TCGA-micro", "METABRIC"],
                            colors = ['blue','lightblue','lightgreen'],
                            boxWidth = 0.2, y_lim = False, no_legend = True):
    # create dataframe
    bic_sizes_df = {}
    i = 0
    for ds in datasets:
        for method in methods :
            for bic in methods_dict[method][ds]:
                if how == "size":
                    res = len(bic[what])
                elif how == "dist":
                    res = bic[what]
                bic_sizes_df[i] = {"method":method,"dataset":ds,what:res}
                i+=1
    bic_sizes_df = pd.DataFrame.from_dict(bic_sizes_df).T
    bic_sizes_df[what] = bic_sizes_df[what].astype(float)
    flierprops = dict(markerfacecolor='grey', markersize=3,marker=".",
              linestyle='none')
    ax = sns.boxplot(x="method", y=what, hue="dataset", hue_order=datasets,
                data=bic_sizes_df, width = boxWidth*len(datasets),
                palette=colors, flierprops=flierprops)
    if no_legend:
        ax.legend_.remove()
    # axis is limited to y_lim
    if y_lim:
        ax.set_ylim(0,min(y_lim, max(bic_sizes_df[what].values)))
    ax.set_xlabel('')
    if how == "size":
        plt.title('number of '+what+' per module')
    elif how == "dist":
        plt.title(what+' per module')

def plot_size_distributions(methods_dict, methods = [],what="samples",
                            datasets = ["TCGA-RNAseq","TCGA-micro", "METABRIC"],
                            colors = ['blue','lightblue','lightgreen'],
                            boxWidth = 0.2, y_lim = 2000):
    # create dataframe
    bic_sizes_df = {}
    i = 0
    for ds in datasets:
        for method in methods :
            for bic in methods_dict[method][ds]:
                bic_sizes_df[i] = {"method":method,"dataset":ds,what:len(bic[what])}
                i+=1
    bic_sizes_df = pd.DataFrame.from_dict(bic_sizes_df).T
    bic_sizes_df[what] = bic_sizes_df[what].astype(float)
    flierprops = dict(markerfacecolor='grey', markersize=3,marker=".",
              linestyle='none')
    ax = sns.boxplot(x="method", y=what, hue="dataset", hue_order=datasets,
                data=bic_sizes_df, width = boxWidth*len(datasets),
                palette=colors, flierprops=flierprops)
    ax.legend_.remove()
    # axis is limited to y_lim
    ax.set_ylim(0,min(y_lim, max(bic_sizes_df[what].values)))
    ax.set_xlabel('')
    plt.title('number of '+what+' per module')


######## Gene set enrichment analysis code ###########
def read_gmt_file(gmt_file,background = False):
    t_0 = time.time()
    g_sets = {}
    all_genes = set()
    with open(gmt_file) as genesets:
        for line in genesets.readlines():
            line = line.rstrip().split("\t")
            n1 = line[0]
            n2 = line[1]
            if background:
                genes= set([x for x in line[2:] if x in background])
            else:
                genes= set([x for x in line[2:] if x != ''])
            all_genes  = all_genes|genes
            g_sets[n1] = genes
    print(gmt_file, "genes:", len(all_genes),"gene sets:",len(g_sets.keys()),file=sys.stderr)
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stderr)
    return all_genes, g_sets

def run_GSOA(bics,all_genes,gene_sets):
    q_vals,names = [], []
    for i in range(0,len(bics)):
        query = bics[i]["genes"]
        neg_log_q_val, name = query_single_gset(query,all_genes=all_genes, gene_sets=gene_sets)
        q_vals.append(neg_log_q_val)
        names.append(name)
    return q_vals, names

def query_single_gset(query, all_genes=[], gene_sets=[]):
    res = calc_pvalues(query, gene_sets, background=all_genes)
    if len(res) == 0:
        return  0, None
    else:
        set_names, p_vals, overlap_size, gset_size, overlapped_genes = res
        # by default Benjamini-Hochberg
        q_vals, rejs = multiple_testing_correction(p_vals)
        #min_log_q_vals.append(np.log10(min(q_vals)))
        best_hit_ndx = np.argmin(q_vals)
        return -np.log10(q_vals[best_hit_ndx]), set_names[best_hit_ndx]

###### Interaction Probability code ########


################# Associations with clinical variables ###########


