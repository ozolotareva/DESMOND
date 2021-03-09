import os
import sys
import time
import copy
import random
import itertools

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from fisher import pvalue


import networkx as nx
import matplotlib.pyplot as plt

from method import identify_opt_sample_set, calc_mean_std_by_powers, remove_small_cc

###################################################### RRHO ########################################################
def relabel_exprs_and_network(exprs,network):
    'Changes gene and sample names to ints'
    g_names2ints  = dict(zip(exprs.index.values, range(0,exprs.shape[0])))
    ints2g_names = exprs.index.values
    s_names2ints = dict(zip(exprs.columns.values, range(0,exprs.shape[1])))  
    ints2s_names = exprs.columns.values
    exprs.rename(index = g_names2ints, columns = s_names2ints,inplace=True)
    network=nx.relabel_nodes(network,g_names2ints)
    return exprs,network, ints2g_names,ints2s_names


def define_SNR_threshold( exprs, exprs_data, network, q,snr_file,
                         min_n_samples=5, random_sample_size= 1000,verbose=True):
    avgSNR = []
    t0 = time.time()
    for edge in random.sample(network.edges(), 1000):
        bic = identify_opt_sample_set([edge[0],edge[1]],exprs,exprs_data,direction="UP",min_n_samples=min_n_samples)
        avgSNR.append(bic["avgSNR"])
    min_SNR = round(np.quantile(avgSNR,q=1-q),3)
    f = open(snr_file, "w")
    print(min_SNR,file = f)
    f.close()
    if verbose:
        print("time:\tMininal avg. |SNR| threshold defined in %s s." %round(time.time()-t0,2), file= sys.stdout)
        print("Mininal avg. |SNR| threshold:\t%s (q=%s)"%(min_SNR,q),file=sys.stdout)
        if os.path.getsize(snr_file) > 0:
            print("Overwrite SNR file:",snr_file,file=sys.stdout)
            return min_SNR


#### Precompute a matrix of thresholds for RRHO ####
def find_threshold(t_u,t_w,N, significance_thr):
    '''Find min. possible overlap still passing the significance threshold, given t_u,t_w and N.'''
    prev_x = N
    for x in range(min(t_u,t_w),0,-1):
        p_val = pvalue(x,t_u-x,t_w-x,N-t_u-t_w+x).right_tail
        #print(t_u,t_w,x, p_val)
        if p_val < significance_thr:
            prev_x = x
        else:
            break
    return prev_x

def precompute_RRHO_thresholds(exprs, fixed_step = 5,significance_thr=0.05):
    t_0 = time.time()
    rrho_thresholds = {}
    for t_u,t_w in itertools.product(range(int(exprs.shape[1]/2),0,-fixed_step), repeat= 2):
        if not t_u in rrho_thresholds.keys():
            rrho_thresholds[t_u] = {}
        rrho_thresholds[t_u][t_w] = find_threshold(t_u,t_w,exprs.shape[1], significance_thr=significance_thr)
    print("Precomputed RRHO thresholds",round(time.time()-t_0,3),"s", file = sys.stdout)
    return rrho_thresholds

#### Perform RRHO ####

def hg_test(rlist1,rlist2,t_u,t_w, return_items = False):
    N = len(rlist1)
    overlap = set(rlist1[:t_u]).intersection(set(rlist2[:t_w]))
    overlap_size = len(overlap)
    p_val = pvalue(overlap_size,t_u-overlap_size,t_w-overlap_size,N+overlap_size-t_u-t_w).right_tail
    enrichment = float(overlap_size)/(float((t_u)*(t_w))/N)
    if return_items:
        return p_val, enrichment, overlap
    else:
        return p_val, enrichment, overlap_size
    
def calc_sums(vals):
    return np.array([len(vals), np.sum(vals), np.sum(np.square(vals))])

def preprocess_power_cumsums(cells_count, inds1, inds2, vals):
    np_sums = np.zeros([cells_count, cells_count, 3])

    np.add.at(np_sums, (inds1, inds2, 0), 1)  
    np.add.at(np_sums, (inds1, inds2, 1), vals)
    np.add.at(np_sums, (inds1, inds2, 2), vals * vals)
    
    reverted = np_sums[::-1, ::-1, :]
    cumsum = reverted.cumsum(axis=0).cumsum(axis=1)
    return cumsum[::-1, ::-1, :]


def calc_SNR(in_powers, all_powers):
    out_powers = all_powers - in_powers
    
    in_mean, in_std = calc_mean_std_by_powers(in_powers)
    out_mean, out_std = calc_mean_std_by_powers(out_powers)
    
    return abs(in_mean - out_mean) / (in_std + out_std)


def more_optimal(count_new, thresh_new, count_old, thresh_old):
    if count_new > count_old:
        return True
    elif count_new == count_old:
        return abs(thresh_new[0] - thresh_new[1]) < abs(thresh_old[0] - thresh_old[1])
    else:
        return False


def partialRRHO(gene1,gene2,subnet,rhho_thrs,min_SNR=0.5,min_n_samples=8,fixed_step=10, verbose = False):
    '''Searches for thresholds corresponding to optimal patient overlap. 
    The overlap is optimal if
    \n1). it passes significance threshold in hypergeometric test;
    \n2). resulting set of samples with SNR higher than 'min_SNR' threshold;
    \n3). its size is maximal.
    \n
    \nReturns a set of samples with gene expressions above selected thresholds.
    '''
    
    t_0 = time.time()
    e1 = subnet.nodes[gene1]["exprs"]
    e1np = np.array(e1.copy().sort_index())

    e2 = subnet.nodes[gene2]["exprs"]
    # e2np = np.array(e2)
    e2np = np.array(e2.copy().sort_index())

    rlist1 = e1.index.values
    inv_rlist1 = np.zeros(len(rlist1), dtype=int) # ind to place in sorted array
    inv_rlist1[rlist1] = np.arange(len(rlist1))
    
    rlist2 = e2.index.values
    inv_rlist2 = np.zeros(len(rlist2), dtype=int)
    inv_rlist2[rlist2] = np.arange(len(rlist2))
    
    assert len(rlist1) == len(rlist2)
    mid = len(rlist1) // 2
    cells_count = (mid + fixed_step - 1) // fixed_step
   
    inds1 = (mid - inv_rlist1 - 1) // fixed_step
    inds2 = (mid - inv_rlist2 - 1) // fixed_step
    good_inds = (inds1 >= 0) & (inds2 >= 0)

    cumsums1 = preprocess_power_cumsums(
        cells_count, inds1[good_inds], inds2[good_inds], 
        vals=e1np[good_inds]
    )
    cumsums2 = preprocess_power_cumsums(
        cells_count, inds1[good_inds], inds2[good_inds], 
        vals=e2np[good_inds]
    )

    arsums1 = calc_sums(e1np)
    arsums2 = calc_sums(e2np)

    optimal_thresholds = (-1, mid)
    optimal_patient_count = 0

    stop_condition = False  # found significant overlap or thresolds reached ends of both lists 
    diag_index = 0
    while not stop_condition:
        # process diag_index'th diag. Calc all points stats
        for du in range(0, diag_index+1):
            dw = diag_index - du 
            tu = mid - fixed_step * du
            tw = mid - fixed_step * dw

            if (tu <= 0) or (tw <= 0):
                stop_condition = True
                break  # why ?

            samples_count = cumsums1[du, dw, 0]
            if samples_count > rhho_thrs[tu][tw]:
                SNR = (calc_SNR(cumsums1[du, dw, :], arsums1) + 
                calc_SNR(cumsums2[du, dw, :], arsums2)) / 2

                if SNR >= min_SNR:
                    stop_condition = True

                    if more_optimal(samples_count, (tu, tw), optimal_patient_count, optimal_thresholds):
                        optimal_thresholds = (tu, tw)
                        optimal_patient_count = samples_count

            if samples_count <= min_n_samples:
                stop_condition = True 

        diag_index += 1
                
    # print(diag_index)
    if optimal_patient_count > 0:
        tu, tw = optimal_thresholds

        optimal_patient_set = set(rlist1[:tu]).intersection(set(rlist2[:tw]))
    else:
        optimal_patient_set = set()
    if verbose:
        if optimal_patient_count > 0 :
            print("runtime:",round(time.time()-t_0,5),"s","Best ovelap for",(gene1,gene2),"occurs at thresholds",optimal_thresholds,"and contains",len(optimal_patient_set),"samples",file = sys.stderr)
        else:
            print("runtime:",round(time.time()-t_0,5),"s","No significant overlap for",(gene1,gene2),file = sys.stderr)
    return optimal_patient_set

#### Assign sets of samples on edges ####
def expression_profiles2nodes(subnet, exprs, direction,verbose = True):
    t_0 = time.time()
    '''Associates sorted expression profiles with nodes. Removes nodes without expressions.
    \If direction="UP" sorts patinets in the descending order, othewise in ascending order - down-regulated first.'''
    if direction == "UP":
        ascending= False
    elif direction == "DOWN":
        ascending= True
    
    # assign expression profiles to every node  
    node2exprs = {}
    # store only sorted sample names
    for gene in exprs.index.values:
        node2exprs[gene] = exprs.loc[gene,:].sort_values(ascending=ascending)
    
    ## set node attributes
    nx.set_node_attributes(subnet, node2exprs, name = 'exprs')
    print("\texpression_profiles2nodes()\truntime:",round(time.time()-t_0,5),"s",file=sys.stdout)
    return subnet

def assign_patients2edges(network,rrho_thrs,min_SNR=0.5,min_n_samples=8, fixed_step=10, verbose=True):
    t0 = time.time() 
    edge2pats = {}
    n_removed,n_retained = 0, 0
    i = 0
    runtimes = []
    for edge in network.edges():
        n1, n2 = edge
        t_0 = time.time()
        samples = partialRRHO(n1,n2,network,rrho_thrs,min_SNR=min_SNR,min_n_samples=min_n_samples,
                            fixed_step=fixed_step,verbose=False)
        runtimes.append(time.time() - t_0)
        i+=1
        if i%1000 == 0 and verbose:
            print("\t",i,"edges processed. Average runtime per edge (s):", round(np.mean(runtimes),5), file=sys.stdout)
        
        # remove edges with not enough samples
        if len(samples)<min_n_samples:
            network.remove_edge(n1,n2)
            n_removed+=1
        else:
            network[n1][n2]["samples"] = samples
            n_retained +=1
            
    
        
    if verbose:
        print("\tassign_patients2edges() runtime:\t", round(time.time() - t0,4),"for subnetwork of",
              len(network.nodes()),"nodes and", len(network.edges()),"edges.", file = sys.stdout)
    if verbose:
        print("Of total %s edges %s retained;\n%s edges containing less than %s samples were removed" % 
          (n_retained+n_removed, n_retained, n_removed, min_n_samples), file=sys.stdout)
    
    n_nodes = len(network.nodes())
    network.remove_nodes_from(list(nx.isolates(network)))
    # get rid of components containing just 1 or 2 nodes
    network = remove_small_cc(network, min_n_nodes = 3)
    
    if verbose:
        print("%s isolated nodes (deg==0) were removed, %s nodes retained." % (n_nodes-len(network.nodes()),len(network.nodes())), file=sys.stdout)
    return network

### plots the distribution of number of samples over all populated edges          
def plot_edge2sample_dist(network,outfile):
    n_samples = []
    for edge in network.edges():
        n1,n2 = edge
        samples = len(network[n1][n2]["samples"])
        n_samples.append(samples)
        # mask edges with not enough samples
    tmp = plt.hist(n_samples,bins=50)
    tmp = plt.title("Distribution of samples associated with edges.")
    plt.savefig(outfile, transparent=True)