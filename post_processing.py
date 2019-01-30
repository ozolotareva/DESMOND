from __future__ import print_function
import sys,os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import time
import seaborn as sns
import copy

from method import bicluster_avg_SNR


############# read trajectories and restore modules ###### 

def check_convergence_condition(edge2module_history,edges_skip_history,min_pletau_steps = 10,max_edges_oscillating =500):
    if len(edge2module_history) <= min_pletau_steps:
        return False
    else:
        skips = edges_skip_history[-min_pletau_steps:]
        #print(min(map(len,skips)) - max(map(len,skips)),map(len,skips))
    if max(map(len,skips)) > max_edges_oscillating:
        return False
    else:
        if max(map(len,skips)) - min(map(len,skips)) <= 0.1* max_edges_oscillating:
            return True
        else:
            return False
        
def restore_modules(edge2module,edges,subnet,exprs):
    t_0 = time.time()
    moduleSizes=np.zeros(len(edge2module))
    all_pats = list(exprs.columns.values)
    nOnesPerPatientInModules= np.zeros((len(edge2module), len(all_pats)))

    for i in range(0,len(edge2module)):
        e2m = edge2module[i]
        # module sizes
        moduleSizes[e2m] +=1
        # number of ones per patient per module
        n1,n2 = edges[i]
        pats_in_module = subnet[n1][n2]["patients"]
        x = np.zeros(len(all_pats), dtype='int')
        i = 0
        for p in all_pats:
            if p in pats_in_module:
                x[i] = 1.0
            i+=1
        nOnesPerPatientInModules[e2m,] += x
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stderr)
    return moduleSizes, nOnesPerPatientInModules

####################### manipulations with modules ##########
def get_genes(mid,edge2module=[],edges=[]):
    ndx = [i for i, j in enumerate(edge2module) if j == mid]
    genes = []
    for edge in [edges[i] for i in ndx]:
        genes.append(edge[0])
        genes.append(edge[1])
    genes = list(set(genes))
    return genes

def get_opt_pat_set(n_ones, m_size, exprs,genes, min_n_patients=50):
    best_pat_set = []
    best_SNR = 0
    best_thr = -1
    thresholds = sorted(list(set(n_ones/m_size)),reverse=True)[:-1]
    thresholds =  [x for x in thresholds if x >= 0.5]
    for thr in thresholds:
        ndx = np.where(n_ones/m_size >= thr)[0]
        pats = exprs.iloc[:,ndx].columns.values
        if len(pats) > min_n_patients:
            avgSNR =  bicluster_avg_SNR(pats=pats, genes=genes, exprs=exprs, absolute = True)
            if avgSNR > best_SNR: 
                best_SNR = avgSNR 
                best_pat_set = pats
                best_thr = thr
    return best_pat_set, best_thr, best_SNR

def plot_bic_stats(n_pats, n_genes, SNRs):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    tmp = plt.hist(n_pats, bins=50)
    tmp = plt.title("n_patients")
    plt.subplot(132)
    tmp = plt.hist(n_genes, bins=50)
    tmp = plt.title("n_genes")
    plt.subplot(133)
    tmp = plt.hist(SNRs, bins=50)
    tmp = plt.title("avg. SNR")  
    
def write_modules(bics,file_name):
    fopen = open(file_name,"w")
    for bic in bics:
        print("id:\t"+str(bic["id"]), file=fopen)
        print("average SNR:\t"+str(bic["SNR"]),file=fopen)
        print("genes:\t"+" ".join(bic["genes"]),file=fopen)
        print("patients:\t"+" ".join(bic["patients"]),file=fopen)
    fopen.close()
    print(str(len(bics)),"modules written to",file_name,file = sys.stderr)
    
    
    
############ merge subnetworks ###############
def add_bic(ndx, ndx2, bics, new_pats, new_SNR, SNRs,pat_overlap, nOnesPerPatientInModules,moduleSizes):
    '''merges bic2 to bic and removes bic2'''
    bic = bics[ndx]
    bic2 = bics[ndx2]
    mid = bic["id"]
    mid2 = bic2["id"]
    # update nOnesPerPatientInModules,moduleSizes by mid
    nOnesPerPatientInModules[mid,:] +=  nOnesPerPatientInModules[mid2,:] 
    moduleSizes[mid] += moduleSizes[mid2]
    nOnesPerPatientInModules[mid2,:] = 0
    moduleSizes[mid2] = 0
    # bic := bic + bic2
    bic["genes"]  =  bic["genes"] | bic2["genes"]
    bic["patients"] =  new_pats
    bic["SNR"] = new_SNR
    # update bic in bics, SNRs,pat_overlap
    bics[ndx] = bic
    SNRs[ndx] = new_SNR
    for j in range(0,len(bics)):
        if j !=ndx:
            shared_genes = len(bic["genes"].intersection(bics[j]["genes"]))
            if shared_genes > 0:
                shared_pats = float(len(bic["patients"].intersection(bics[j]["patients"])))
                mutual_overlap = min(shared_pats/len(bic["patients"]),shared_pats/len(bics[j]["patients"]))
                pat_overlap[ndx,j] = mutual_overlap 
                pat_overlap[j,ndx] = mutual_overlap
    # remove bic2
    bics, SNRs,pat_overlap = remove_bic(ndx2, bics, SNRs,pat_overlap)
    return bics, SNRs, pat_overlap, nOnesPerPatientInModules,moduleSizes

def remove_bic(ndx, bics, SNRs,pat_overlap):
    bic = bics[ndx]
    mid = bic["id"]
    # from bics, SNRs,pat_overlap by ndx
    bics = bics[:ndx] +  bics[ndx+1:]
    SNRs = SNRs[:ndx] +  SNRs[ndx+1:]
    pat_overlap = np.delete(pat_overlap, ndx, axis=1)
    pat_overlap = np.delete(pat_overlap, ndx, axis=0)
    return bics, SNRs,pat_overlap

def calc_new_SNR(bic, bic2,exprs, nOnesPerPatientInModules,moduleSizes,min_n_patients=50):
    mid = bic["id"]
    mid2 = bic2["id"]
    n_ones = nOnesPerPatientInModules[mid,]+nOnesPerPatientInModules[mid2,]
    m_size = moduleSizes[mid,]+moduleSizes[mid2,]
    genes  = bic["genes"] | bic2["genes"]
    pats, thr, avgSNR = get_opt_pat_set(n_ones, m_size, exprs, genes, min_n_patients=50)
    return pats, avgSNR


def merge_modules(bics,nOnesPerPatientInModules,moduleSizes,exprs,SNRs = [], 
                 min_patient_overlap = 0.5,min_acceptable_SNR_percent=0.9, verbose = True):
    t_0 = time.time()
    if len(SNRs) == 0:
        SNRs = [bic["SNR"] for bic in bics]
    if verbose:
        print("Input:", len(bics),"modules to merge", file = sys.stderr)
    
    ############## prepare patient overlap matrix  #########
    pat_overlap = np.zeros((len(bics),len(bics)))
    gene_overlap = np.zeros((len(bics),len(bics)))
    for i in range(0,len(bics)):
        bic_i = bics[i]
        for j in range(0,len(bics)):
            if i !=j:
                bic_j = bics[j]
                shared_genes = len(bic_i["genes"].intersection(bic_j["genes"]))
                gene_overlap[i,j] = shared_genes 
                if shared_genes >0:
                    shared_pats = float(len(bic_i["patients"].intersection(bic_j["patients"])))
                    mutual_overlap = min(shared_pats/len(bic_i["patients"]),shared_pats/len(bic_j["patients"]))
                    pat_overlap[i,j] = mutual_overlap 
    
    ############ run merging #############
    closed_modules = []
    while len(bics) > 0:
        ndx = np.argmax(SNRs)
        bic = bics[ndx]
        if verbose:
            print("Grow module:",bic["id"],"SNR",bic["SNR"], "pats",len(bic["patients"]), "genes",  bic["genes"])
        best_candidate_ndx = -1
        best_new_SNR = min_acceptable_SNR_percent*SNRs[ndx]
        best_pat_set = bic["patients"]
        candidate_ndxs = np.where(pat_overlap[ndx,:] >= min_patient_overlap )[0]
        for candidate_ndx in candidate_ndxs:
            bic2 = bics[candidate_ndx]
            if verbose:
                print("\t trying module:",bic2["id"],"SNR",bic2["SNR"], "pats",len(bic2["patients"]), "genes", bic2["genes"])
            new_pats, new_SNR = calc_new_SNR(bic, bic2 ,exprs, nOnesPerPatientInModules,moduleSizes)
            if verbose:
                print("\t\t SNR:",new_SNR,"passed:",new_SNR > best_new_SNR)
            if new_SNR > best_new_SNR:
                #print("\t\t set SNR:", best_new_SNR, "-->", new_SNR, "and select",candidate_ndx)
                best_new_SNR = new_SNR
                best_candidate_ndx  =  candidate_ndx
                best_pat_set = new_pats
        if best_candidate_ndx >= 0:
            if verbose:
                print("\tmerge",bic["id"],"+",bics[best_candidate_ndx]["id"], "SNR",best_new_SNR,"patients",len(best_pat_set))
            # add bic2 to bic and remove bic2
            bics, SNRs, pat_overlap, nOnesPerPatientInModules, moduleSizes = add_bic(ndx, best_candidate_ndx,bics,
                                                                                             set(best_pat_set), best_new_SNR,
                                                                                             SNRs,pat_overlap, 
                                                                                             nOnesPerPatientInModules,
                                                                                             moduleSizes)
            # continue - pick new ndx_max
        else:
            if verbose:
                print("no more candidates to merge for ", bic["id"])
            # close module if no candidates for merging found
            closed_modules.append(bic)
            if verbose:
                print("----------- closed. ", bic["id"])
            # remove module from all data structures
            bics, SNRs,pat_overlap = remove_bic(ndx, bics, SNRs, pat_overlap)
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stderr)
    return closed_modules
