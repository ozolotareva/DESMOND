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


from method import bicluster_avg_SNR, merge_biclusters
from desmond_io import read_bic_table

#####  read resulted biclusters #######

def read_bicge(bicge_file, exprs,verbose = True):
    bics = {}
    i=0
    n_zero_bics = 0 
    with open(bicge_file,"r") as infile:
        for line in infile.readlines():
            line = line.rstrip().split()
            if len(line)>0:
                if line[0] == 'Gene_names:':
                    i+=1
                    bic = {}
                    genes = set(line[1:])
                    bic["genes"] = genes
                    bic["n_genes"]  = len(genes)
                if line[0] == 'Condition_names:':
                    samples = set(line[1:])
                    #bic["id"] = i
                    bic["samples"] = samples
                    bic["n_samples"]  = len(samples)
                    bic["avgSNR"] = bicluster_avg_SNR(exprs, samples=samples, genes = genes)
                    if bic["n_samples"] > 0 and bic["n_genes"]>0:
                        bics[i] = bic
                    else:
                        n_zero_bics +=1 
    if verbose and n_zero_bics>0:
        print("\tbiclusters with no patients of genes:", n_zero_bics)
    bics = pd.DataFrame.from_dict(bics).T
    bics = bics[["n_genes","n_samples","avgSNR","genes","samples"]]
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

    
################# Reproducibility ################################

def find_bm_genes(df_target, df_db):
    target = df_target.T.to_dict()
    db = df_db.T.to_dict()
    BMs = {} 
    for i in target.keys():
        bm_j, best_J = -1, 0
        bic = target[i]
        for j in db.keys():
            bic2 = db[j]
            J = len(bic["genes"].intersection(bic2["genes"]))*1.0/len(bic["genes"].union(bic2["genes"]))
            if J > best_J:
                best_J = J
                bm_j = j
        BMs[i] = {"bm_id":bm_j,"J":best_J}
    BMs = pd.DataFrame.from_dict(BMs).T
    BMs.loc[:,"bm_id"] = BMs.loc[:,"bm_id"].apply(int)
    return BMs

def compare_TCGA_and_METABRIC_GeneSets(results_dict,verbose = True,min_SNR=0,
                                       order=["DeBi","ISA","FABIA","COALESCE","QUBIC","DESMOND"]):
    dfs = []
    for method in [k for k in results_dict.keys() if "DeBi" not in k]:
        if verbose:
            print(method)
        try:
            Tm = results_dict[method]["TCGA-micro"]
            Tr = results_dict[method]["TCGA-RNAseq"]
            M = results_dict[method]["METABRIC"]
            Tm = Tm.loc[Tm["avgSNR"]>=min_SNR,:]
            Tr = Tr.loc[Tr["avgSNR"]>=min_SNR,:]
            M = M.loc[M["avgSNR"]>=min_SNR,:]
            print("TCGA-micro:", Tm.shape[0], "TCGA-RNAseq:",Tr.shape[0],"METABRIC:",M.shape[0])
            if Tm.shape[0]>0 and Tr.shape[0]>0 and M.shape[0]>0:
                passed =True
            else:
                print("not enough data for",method)
                passed = False
        except:
            print("\tnot found data for", method)
            passed = False
        if passed:
            Tm_M = pd.DataFrame(list(find_bm_genes(Tm,M)["J"].values) + list(find_bm_genes(M,Tm)["J"].values))
            Tm_M.columns = ["Jg"]
            Tm_M["datasets"] = "TCGA-micro vs METABRIC"
            Tm_M.index = range(0,Tm_M.shape[0])
            Tr_M = pd.DataFrame(list(find_bm_genes(Tr,M)["J"].values) + list(find_bm_genes(M,Tr)["J"].values))
            Tr_M.columns = ["Jg"]
            Tr_M["datasets"] = "TCGA-RNAseq vs METABRIC"
            Tr_M.index = range(Tm_M.shape[0]+1,Tm_M.shape[0]+1+ Tr_M.shape[0])
            
            Tr_Tm = pd.DataFrame(list(find_bm_genes(Tr,Tm)["J"].values) + list(find_bm_genes(Tm,Tr)["J"].values))
            Tr_Tm.columns = ["Jg"]
            Tr_Tm["datasets"] = "TCGA-micro vs TCGA-RNAseq"
            Tr_Tm.index = range(Tm_M.shape[0]+Tr_M.shape[0]+1,Tm_M.shape[0]+Tr_M.shape[0]+1+ Tr_Tm.shape[0])
            
            df = pd.concat([Tm_M,Tr_M,Tr_Tm],axis =0)
            if "-D" in method:
                m = method.replace("-D","")
                df["parameters"] = "Default"
            else:
                m = method.replace("-T","")
                df["parameters"] = "Tuned"
            df["method"] = m
            dfs.append(df)
    dfs = pd.concat(dfs,axis =0)
    dfs2 =[]
    for method in order:
        dfs2.append(dfs.loc[dfs["method"].str.contains(method),:])
    dfs2 = pd.concat(dfs2,axis =0)
    return dfs2


def find_bm_biclusters(df_target, df_db, shared_samples = set()):
    target = df_target.T.to_dict()
    db = df_db.T.to_dict()
    BMs = {} 
    for i in target.keys():
        bm_j, best_J = -1, 0
        bic = target[i]
        for j in db.keys():
            bic2 = db[j]
            Jg = len(bic["genes"].intersection(bic2["genes"]))*1.0/len(bic["genes"].union(bic2["genes"]))
            # !consider only shared samples for Js
            Js = len(bic["samples"].intersection(bic2["samples"]))*1.0/len(bic["samples"].union(bic2["samples"]).intersection(shared_samples))
            J = Jg*Js
            if Jg*Js > best_J:
                best_J = J
                bm_j = j
        BMs[i] = {"bm_id":bm_j,"J":best_J}
    BMs = pd.DataFrame.from_dict(BMs).T
    BMs.loc[:,"bm_id"] = BMs.loc[:,"bm_id"].apply(int)
    return BMs

def compare_TCGA_biclusters(results_dict, shared_samples,verbose = True, min_SNR=0,
                           order=["DeBi","ISA","FABIA","COALESCE","QUBIC","DESMOND"]):
    dfs = []
    for method in [k for k in results_dict.keys() if "DeBi" not in k]:
        try:
            Tm = results_dict[method]["TCGA-micro"]
            Tm = Tm.loc[Tm["avgSNR"]>=min_SNR,:]
            Tr = results_dict[method]["TCGA-RNAseq"]
            Tr = Tr.loc[Tr["avgSNR"]>=min_SNR,:]
            print(method, "TCGA-micro:",Tm.shape[0],  "TCGA-RNAseq:", Tr.shape[0])
            if Tm.shape[0]>0 and Tr.shape[0]>0:
                passed =True
            else:
                print("not enough data for",method)
                passed = False
        except:
            print("not enough data for", method)
            passed = False
        if passed:
            
            r2m = find_bm_biclusters(Tr, Tm, shared_samples = shared_samples)
            r2m["datasets"] = "TCGA-RNAseq vs TCGA-micro" 
            m2r =  find_bm_biclusters(Tm, Tr, shared_samples = shared_samples)
            m2r["datasets"] = "TCGA-micro vs TCGA-RNAseq" 
            df = pd.concat([r2m, m2r], axis =0)
            if "-D" in method:
                m = method.replace("-D","")
                df["parameters"] = "Default"
            else:
                m = method.replace("-T","")
                df["parameters"] = "Tuned"
            df["method"] = m
            dfs.append(df)
    dfs = pd.concat(dfs,axis =0 )
    dfs2 =[]
    for method in order:
        dfs2.append(dfs.loc[dfs["method"].str.contains(method),:])
    dfs2 = pd.concat(dfs2,axis =0)
    return dfs2
    
################# Associations with clinical variables ###########

def hypergeom_test(patients,anno_dict,anno_pats,categories=[],sign_thr =0.05):
    # e.g. whether bicluster membership is assicated with group membership
    in_bicluster = set(patients).intersection(anno_pats)
    outside_bicluster =  anno_pats.difference(set(patients))
    best_p_val = 0.05
    enriched_cat = "NA"
    best_fold_enrichment = 0
    best_overlap = 0
    if len(categories) == 0:
        categories = anno_dict.keys()
    for category in categories:
        in_group = anno_dict[category]
        outside_group = anno_pats.difference(in_group)
        #print(field, category,len(in_bicluster), len(outside_bicluster), len(in_group), len(outside_group))
        # define group membership
        overlap = len(in_bicluster.intersection(in_group))
        outside_both = len(outside_bicluster.intersection(outside_group))
        in_bicluster_outside_group = len(in_bicluster.intersection(outside_group))
        outside_bicluster_in_group = len(set(outside_bicluster).intersection(set(in_group)))
        # right-sided exact Fisher's test
        p_val = pvalue(overlap,in_bicluster_outside_group,outside_bicluster_in_group,outside_both).right_tail
        
        if p_val < 0.05:
            expected_overlap = float(len(in_group))/len(anno_pats)*len(in_bicluster)
            fold_enrichment = float(overlap)/expected_overlap
            #print(p_val, category)
            log_neg_pval = -np.log10(p_val)
            if best_p_val < log_neg_pval:
                best_p_val = log_neg_pval
                enriched_cat = category
                best_fold_enrichment = fold_enrichment
                best_overlap = overlap
    return best_p_val,best_fold_enrichment,best_overlap, enriched_cat 

def apply_hypergeom(bics, annotation, field="mol_subt", title="subtitle",categories=[]):
    # prepare annotation-convert to dict
    anno = annotation.loc[[field],:].T
    anno.dropna(inplace = True)
    anno_pats =  set(anno[field].index)
    #print(anno.shape)
    anno_dict = {}
    for cat in categories:
        in_group = set(anno.loc[anno[field]==cat,:].index)
        # if size of group is more than |P|/2 redefine the group
        if len(in_group) > len(anno_pats)/2:
            in_group= anno_pats.difference(in_group)
        anno_dict[cat] = in_group 
        
    # make dataframe 
    df = {}
    for i in bics.keys():
        samples = bics[i]["samples"]
        lnp, fc, overlap, cat = hypergeom_test(samples,anno_dict,anno_pats,
                                  categories=categories,
                                  sign_thr =0.05)
        df[i] = {field:cat,"-log10(p-value)":lnp,"Fold_Enrich":fc,"overlap":overlap,"samples_in_category":len(anno_dict[cat])}
        df[i].update(bics[i])
    df = pd.DataFrame.from_dict(df).T
    df["-log10(p-value)"] = df["-log10(p-value)"].apply(float)
    df["J"] = df["overlap"]/(df["samples_in_category"]+ df["n_samples"]- df["overlap"])
    df = apply_BH(df, alpha = 0.05)
    
    return df

def apply_BH(df, alpha = 0.05):
    df.loc[:,"ndx"] = range(0,df.shape[0])
    df.loc[:,"ndx"] += 1 
    df.loc[:,"p_val_adj"] = df["ndx"]*alpha/6/df.shape[0]
    df.loc[:,"p_vals"] = 10**(-1*df["-log10(p-value)"])
    df.loc[:,"passed_BH"] = (df["p_vals"] < df["p_val_adj"] )
    df.loc[(df["p_vals"] >= df["p_val_adj"] ),"mol_subt"] = "NA"
    return df


########################### GO Overlap Analysis #######################
def run_GSOA(bics,all_genes,gene_sets):
    q_vals,names, FCs = [], [], []
    for i in range(0,len(bics)):
        query = bics[i]["genes"]
        neg_log_q_val, name, FC = query_single_gset(query,all_genes=all_genes, gene_sets=gene_sets)
        q_vals.append(neg_log_q_val)
        names.append(name)
        FCs.append(FC)
    return q_vals, names, FCs

def query_single_gset(query, all_genes=[], gene_sets=[]):
    res = calc_pvalues(query, gene_sets, background=all_genes)
    if len(res) == 0:
        return  0, None,0
    else:
        set_names, p_vals, overlap_size, gset_size, overlapped_genes = res
        # by default Benjamini-Hochberg
        q_vals, rejs = multiple_testing_correction(p_vals)
        #min_log_q_vals.append(np.log10(min(q_vals)))
        best_fc = 0
        best_hit_ndx = -1
        for i in range(0,len(q_vals)):
            if q_vals[i]<0.05:
                obs = float(overlap_size[i])
                expec = float(gset_size[i])*len(query)/len(all_genes)
                fc = obs/expec
                if fc > best_fc:
                    best_fc=fc
                    best_hit_ndx = i
        if best_hit_ndx == -1:
            return  0, None,0
        else:
            return -np.log10(q_vals[best_hit_ndx]), set_names[best_hit_ndx], np.log10(best_fc)

def plot_perc_enrihment(dbs_dict,
                        datasets = ["TCGA-RNAseq","TCGA-micro","TCGA-micro"],
                        methods = [],
                        dbs= ["GOBP", "GOMF","CC"],
                        colors = ['green','red',"blue"], 
                        barWidth = 0.2, title='Gene Set Overlap Analysis results', legend = True):
    signif_hits = {}
    for db_name in dbs:
        signif_hits[db_name] = []
        for method in methods:
            q_vals = []
            for ds in datasets:
                q_vals += dbs_dict[db_name][ds][method][0] 
            try:
                perc_signif = len([x for x in q_vals if x != 0])*100.0/len(q_vals) # % significant hits 
            except :
                perc_signif = 0
            #print(db_name,method,ds, round(perc_signif/100,2),len(q_vals))
            signif_hits[db_name].append(perc_signif)
    
    # The x position of bars
    # first position
    bar_positions = [np.arange(len(signif_hits[db_name]))]
    # from 1 to n
    for i in range(1, len(dbs)):
        bar_positions.append( [x + barWidth for x in bar_positions[i-1]])

    # Create  bars
    for i in range(len(dbs)):
        plt.bar(bar_positions[i],  signif_hits[dbs[i]], width = barWidth, color = colors[i],
                edgecolor = 'grey', capsize=7, label=dbs[i])

    # general layout
    plt.xticks([r + barWidth for r in range(len(signif_hits[dbs[0]]))], methods,rotation=30)
    plt.title(title)
    plt.ylabel('% of gene clusters significantly overlapped\n with at least one gene set')
    if legend:
        plt.legend() # loc='upper left'