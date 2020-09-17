from __future__ import print_function
import os,sys
import time
import warnings
import time
import copy
import random

import pandas as pd
import numpy as np
from fisher import pvalue
import itertools
import math

from sklearn.cluster import KMeans

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt 

###################################################### RRHO ####################################################################

def define_SNR_threshold(snr_file, exprs, exprs_data, network, q,
                         min_n_samples=5, random_sample_size= 1000,verbose=True):
    if os.path.exists(snr_file) and os.path.getsize(snr_file) > 0:
        try:
            f = open(snr_file,"r")
            min_SNR = float(f.readlines()[0])
            f.close()
            if verbose:
                print("Use pre-computed SNR threshold.", file= sys.stdout)
            return (min_SNR)
        except:
            pass
        
    avgSNR = []
    t0 = time.time()
    for edge in random.sample(network.edges(), 1000):
        bic = identify_opt_sample_set([edge[0],edge[1]],exprs,exprs_data,direction="UP",min_n_samples=min_n_samples)
        if bic >0:
            avgSNR.append(bic["avgSNR"])
        else:
            avgSNR.append(0)
    min_SNR = np.quantile(avgSNR,q=1-q)
    f = open(snr_file, "w")
    print(round(min_SNR,2),file = f)
    f.close()
    if verbose:
        print("time:\tSNR threshold determined in %s s." %round(time.time()-t0,2), file= sys.stdout)
    return round(min_SNR,2)

def relabel_exprs_and_network(exprs,network):
    'Changes gene and sample names to ints'
    g_names2ints  = dict(zip(exprs.index.values, range(0,exprs.shape[0])))
    ints2g_names = exprs.index.values
    s_names2ints = dict(zip(exprs.columns.values, range(0,exprs.shape[1])))  
    ints2s_names = exprs.columns.values
    exprs.rename(index = g_names2ints, columns = s_names2ints,inplace=True)
    network=nx.relabel_nodes(network,g_names2ints)
    return exprs,network, ints2g_names,ints2s_names

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
    for t_u,t_w in itertools.product(range(exprs.shape[1]/2,0,-fixed_step), repeat= 2):
        if not t_u in rrho_thresholds.keys():
            rrho_thresholds[t_u] = {}
        rrho_thresholds[t_u][t_w] = find_threshold(t_u,t_w,exprs.shape[1], significance_thr=significance_thr)
    print("Precomputed RRHO thresholds",time.time()-t_0,"s", file = sys.stdout)
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


def calc_mean_std_by_powers(powers):
    count, val_sum, sum_sq = powers

    mean = val_sum / count  # what if count == 0?
    std = np.sqrt((sum_sq / count) - mean*mean)
    return mean, std


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
    e1 = subnet.node[gene1]["exprs"]
    e1np = np.array(e1.copy().sort_index())

    e2 = subnet.node[gene2]["exprs"]
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
    nx.set_node_attributes(subnet, 'exprs', node2exprs)
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
            print("\t",i,"edges processed. Average runtime per edge:",np.mean(runtimes), file=sys.stdout)
        
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
    if verbose:
        print("%s isolated nodes (deg==0) were removed, %s nodes retained." % (n_nodes-len(network.nodes()),len(network.nodes())), file=sys.stdout)
    return network

#################################### Gibbs Sampling #######################################################

def calc_lp(edge,self_module,module,edge2Patients,
            nOnesPerPatientInModules,moduleSizes,
            moduleOneFreqs,p0,match_score,mismatch_score,bK_1,
            alpha=1.0,beta_K=1.0):
    
    N = edge2Patients.shape[1]
    m_size = moduleSizes[module]
    edge_vector = edge2Patients[edge,]
    
  
    if m_size == 0:
        return p0
    
    n_ones_per_pat = nOnesPerPatientInModules[module,]
    
    if self_module == module: # remove the edge from the module, if it belongs to a module
        if m_size == 1:
            return p0
        m_size -=1
        n_ones_per_pat = n_ones_per_pat-edge_vector
    
    # if a module is composed of a single edge
    if m_size == 1:
        # just count number of matches and mismatches and
        n_matches =  np.inner(n_ones_per_pat,edge_vector)
        return n_matches*match_score+(N-n_matches)*mismatch_score + bK_1
    
    # if a module contains more than one edge
    beta_term = math.log(m_size+beta_K)
    
    # alpha_term
    # ones-matching
    oneRatios = (n_ones_per_pat+alpha/2)/(m_size+alpha)
    ones_matching_term = np.inner(np.log(oneRatios),edge_vector)
    # zero-matching
    zeroRatios = (m_size-n_ones_per_pat+alpha/2)/(m_size+alpha)
    zeros_matching_term = np.inner(np.log(zeroRatios),(1-edge_vector))

    return ones_matching_term+zeros_matching_term + beta_term

def set_initial_conditions(network, p0, match_score, mismatch_score, bK_1, N, alpha = 1.0,
                           beta_K = 1.0,verbose = True):
    t_0 = time.time()
    
    # 1. the number of edges inside each component, initially 1 for each component
    moduleSizes=np.ones(len(network.edges()),dtype=np.int)
    
    # 2. a binary (int) matrix of size n by m that indicates the samples on the edges
    edge2Patients = []
    all_samples = range(N)
    for edge in network.edges():
        n1,n2 = edge
        samples_in_module = network[n1][n2]["samples"]
        x = np.zeros(len(all_samples), dtype=np.int)
        i = 0
        for p in all_samples:
            if p in samples_in_module:
                x[i] = 1
            i+=1
        edge2Patients.append(x)

    edge2Patients = np.asarray(edge2Patients)
    
    # 3. a binary matrix of size K by m that stores the total number of ones per patient in each module,
    # initially equal to 'edge2Patients'
    nOnesPerPatientInModules = copy.copy(edge2Patients)

    t_0 = time.time()
    i = 0
    for n1,n2,data in network.edges(data=True):
        #del data['masked']
        del data['samples']
        data['m'] = i
        data['e'] = i
        i+=1
    #4.
    edge2Module = range(0,len(network.edges()))
    
    #5. moduleOneFreqs
    moduleOneFreqs = []
    n = edge2Patients.shape[0]
    for e in range(0,n):
        moduleOneFreqs.append(float(sum(edge2Patients[e,]))/edge2Patients.shape[1])
    
    #6. setting initial LPs
    t_0 = time.time()
    t_1=t_0
    for n1,n2,data in network.edges(data=True):
        m = data['m']
        e = data['e']
        data['log_p'] = []
        data['modules'] = []
        for n in [n1,n2]:
            for n3 in network[n].keys():
                m2 = network[n][n3]['m']
                if not m2 in data['modules']:
                    lp = calc_lp(e,m,m2,edge2Patients,
                                 nOnesPerPatientInModules,moduleSizes,
                                 moduleOneFreqs,p0, match_score,mismatch_score,
                                 bK_1, alpha=alpha, beta_K=beta_K)
                    data['log_p'].append(lp)
                    data['modules'].append(m2)
    print("time:\tInitial state created in",round(time.time()- t_0,1) , "s.", file = sys.stdout)
    return moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs, network

def adjust_lp(log_probs,n_exp_orders=7):
    # adjusting the log values before normalization to avoid under-flow
    max_p = max(log_probs)
    probs = []
    for lp in log_probs:
        # shift all probs to set max_prob less than log(max_np.float)  
        adj_lp = lp - max_p
        # set to minimal values all which less then 'n_orders' lesser than p_max to zeroes
        if adj_lp >= - n_exp_orders:
            probs.append(np.exp(adj_lp))
        else:
            probs.append(0)
    probs = probs/sum(probs)
    return probs

### functions for checking of convergence conditions ###
def calc_p_transitions(states,unique,counts):
    n_steps = len(states)-1
    transitions = dict(zip(tuple(itertools.product(unique,unique)),np.zeros(len(unique)**2)))
    for i in range(0,n_steps):
        transitions[(states[i],states[i+1])] += 1 
    p = { k:v/(counts[unique.index(k[0])]) for k, v in transitions.iteritems()}
    return  p

def collect_all_p(labels):
    P={}
    # calculate edge transition probabilities
    for edge in range(0,labels.shape[1]):
        states = labels[:,edge]
        unique,counts = np.unique(states , return_counts=True)
        if len(unique)> 1:
            P[edge] = calc_p_transitions(states,list(unique),counts)
    return P

def calc_RMSD(P,P_prev):
    t0 = time.time()
    p_prev_edges = set(P_prev.keys())
    p_edges = set(P.keys())
    Pdiff = []
    for edge in p_edges.difference(p_prev_edges):
        P_prev[edge] = {k:0 for k in P[edge].keys()}
        P_prev[edge] = {k:1 for k in P_prev[edge].keys() if k[0]==k[1]}
    for edge in p_prev_edges.difference(p_edges):
        P[edge] = {k:0 for k in P_prev[edge].keys()}
        P[edge] = {k:1 for k in P[edge].keys() if k[0]==k[1]}
    for edge in p_edges.intersection(p_prev_edges):
        p_modules = set(P[edge].keys())
        p_prev_modules = set(P_prev[edge].keys())
        for  m,m2 in p_modules.difference(p_prev_modules):
            Pdiff.append((P[edge][(m,m2)])**2) 
        for  m,m2 in p_prev_modules.difference(p_modules):
            Pdiff.append((P_prev[edge][(m,m2)])**2) 
        for  m,m2 in p_modules.intersection(p_prev_modules):
            Pdiff.append((P[edge][(m,m2)] - P_prev[edge][(m,m2)])**2) 
    if not len(Pdiff)==0:
        return np.sqrt(sum(Pdiff)/len(Pdiff))
    else:
        return 0

def check_convergence_conditions(n_skipping_edges,n_skipping_edges_range,
                                P_diffs,P_diffs_range,step,tol=0.05, verbose = True):
    n_points = len(n_skipping_edges)
    # check skipping edges 
    se_min, se_max = n_skipping_edges_range
    n_skipping_edges = np.array(n_skipping_edges,dtype=float)

    
    # scale
    n_skipping_edges = (n_skipping_edges-se_min)/(se_max - se_min)*n_points
    # fit line

    A = np.vstack([range(0,n_points), np.ones(n_points)]).T
    k,b = np.linalg.lstsq(A, n_skipping_edges, rcond=None)[0]
    
    # check P_diffs
    P_diffs_min, P_diffs_max = P_diffs_range
    P_diffs = np.array(P_diffs)
    
    # scale 
    P_diffs = (P_diffs-P_diffs_min)/(P_diffs_max- P_diffs_min)*n_points
    k2, b2  = np.linalg.lstsq(A, P_diffs, rcond=None)[0]
    if abs(k)<tol and abs(k2)<tol:
        convergence = True
    else:
        convergence = False
    if verbose:
        print("\tConverged:",convergence,"#skipping edges slope:",round(k,5),
          "RMS(Pn-Pn+1) slope:",round(k2,5))
    return convergence     
        
### sample and update model when necessary 
def sampling(network, edge2Module, edge2Patients,nOnesPerPatientInModules,moduleSizes,
             moduleOneFreqs, p0, match_score, mismatch_score, bK_1, alpha = 0.1, beta_K = 1.0,
             max_n_steps=100,n_steps_averaged = 20,n_points_fit = 10,tol = 0.1,
             n_steps_for_convergence = 5,verbose=True, edge_ordering ="nosort"):
    edge_order = range(0, edge2Patients.shape[0])
    if edge_ordering =="shuffle":
        # shuffle edges
        random.shuffle(edge_order)
    t_ =  time.time()
    edge2Module_history = [copy.copy(edge2Module)]
    is_converged = False
    network_edges = network.edges(data=True)
    for step in range(1,max_n_steps):
        if verbose:
            print("step",step,file = sys.stdout)
        not_changed_edges = 0
        t_0 = time.time()
        t_1=t_0
        i = 1
        for edge_index in edge_order:
            n1,n2,data = network_edges[edge_index]
            # adjust LogP and sample a new module
            p = adjust_lp(data['log_p'],n_exp_orders=7)
            curr_module = data['m']
            edge_ndx = data['e']
            new_module = np.random.choice(data['modules'], p = p) 

            # update network and matrices if necessary
            if new_module != curr_module:
                apply_changes(network,n1,n2,edge_ndx,curr_module,new_module,
                              edge2Patients, edge2Module, nOnesPerPatientInModules,moduleSizes,
                              moduleOneFreqs, p0,match_score,mismatch_score, bK_1,
                              alpha=alpha,beta_K=beta_K)
                
            else:
                not_changed_edges +=1#
            i+=1
            if i%10000 == 1:
                if verbose:
                    print(i,"\t\tedges processed in",round(time.time()- t_1,1) , "s runtime...",file=sys.stdout)
                not_changed_edges=0
                t_1 = time.time()
        if verbose:
            print("\tstep ",step,# 1.0*not_changed_edges/len(edge_order),"- % edges not changed; runtime",
                  round(time.time()- t_0,1) , "s", file = sys.stdout)
        
        edge2Module_history.append(copy.copy(edge2Module))
        if step == n_steps_averaged:
            is_converged = False
            n_times_cc_fulfilled = 0
            labels = np.asarray(edge2Module_history[step-n_steps_averaged:step])
            P_prev = collect_all_p(labels)
            P_diffs = []
            n_skipping_edges = [] 
            n_skipping_edges.append(len(P_prev.keys()))
        if step > n_steps_averaged:
            labels = np.asarray(edge2Module_history[step-n_steps_averaged:step])
            P = collect_all_p(labels)
            P_diff = calc_RMSD(copy.copy(P),copy.copy(P_prev))
            P_diffs.append(P_diff)
            n_skipping_edges.append(len(P.keys()))
            P_prev=P
        if  step >= n_steps_averaged + n_points_fit:
            P_diffs_range = min(P_diffs),max(P_diffs)
            n_skipping_edges_range= min(n_skipping_edges), max(n_skipping_edges)
            # check convergence condition
            is_converged = check_convergence_conditions(n_skipping_edges[-n_points_fit:],
                                                      n_skipping_edges_range,
                                                      P_diffs[-n_points_fit:],
                                                      P_diffs_range,
                                                      step,
                                                      tol=tol,
                                                      verbose = verbose)
        if is_converged:
            n_times_cc_fulfilled +=1
        else:
            n_times_cc_fulfilled = 0
            
        if n_times_cc_fulfilled == n_steps_for_convergence: # stop if convergence is True for the last n steps
            ### define how many the last steps to consider
            n_final_steps = n_points_fit+n_steps_for_convergence
            if verbose:
                print("The model converged after", step,"steps.", file = sys.stdout)
                print("Consensus of last",n_final_steps,"states will be taken")
                print("Sampling runtime",round(time.time()- t_ ,1) , "s", file = sys.stdout)
            return edge2Module_history, n_final_steps,n_skipping_edges,P_diffs
    
    n_final_steps = n_steps_for_convergence
    if verbose:
        print("The model did not converge after", step,"steps.", file = sys.stdout)
        print("Consensus of last",n_final_steps,"states will be taken")
        print("Sampling runtime",round(time.time()- t_ ,1) , "s", file = sys.stdout)
        
    return edge2Module_history,n_final_steps,n_skipping_edges,P_diffs

def apply_changes(network,n1,n2, edge_ndx,curr_module, new_module,
                  edge2Patients,edge2Module,nOnesPerPatientInModules,moduleSizes,
                  moduleOneFreqs,p0,match_score,mismatch_score,bK_1,
                  alpha=1.0,beta_K=1.0,no_LP_update = False):
    '''Moves the edge from current module to the new one
    and updates network, nOnesPerPatientInModules and moduleSizes respectively.'''
    # update the edge module membership

    network[n1][n2]['m'] = new_module
    edge2Module[edge_ndx] = new_module
    # for this edge no probabilities change
    
    # reduce curr_module size and nOnesPerPatientInModules
    edge_vector = edge2Patients[edge_ndx,]
    nOnesPerPatientInModules[curr_module,] = nOnesPerPatientInModules[curr_module,] - edge_vector
    moduleSizes[curr_module,]-=1
    
    # increase new_module
    nOnesPerPatientInModules[new_module,] = nOnesPerPatientInModules[new_module,] + edge_vector
    moduleSizes[new_module,]+=1
    
    # update LPs for all edges contacting curr and new modules, except skipping edge
    # for affected edges, calcualte only probabilities regarding curr and new modules
    if not no_LP_update:
        for n in [n1,n2]:
            for n3 in network[n].keys():
                if not n3 == n1 and not n3==n2: # for 
                    data = network[n][n3]
                    m = data['m']
                    e = data['e']
                    #### update LP for new_module
                    lp = calc_lp(e,m,new_module,edge2Patients,nOnesPerPatientInModules,moduleSizes,
                                 moduleOneFreqs,p0,match_score,mismatch_score,bK_1,
                                 alpha=alpha,beta_K=beta_K)
                    # find index of targe module or add it if did not exist
                    if new_module in data['modules']:
                        m_ndx = data['modules'].index(new_module)
                        data['log_p'][m_ndx] = lp
                        
                    else: # if it is a novel connection to a new_module, append it to the end of a list
                        data['modules'].append(new_module)
                        data['log_p'].append(lp)
                        
                    #### update LP for curr_module
                    # check if an edge is still connected with curr_module
                    still_connected = False
                    # iterate all edges adjacent to affected
                    for n1_,n2_,data_ in network.edges([n,n3],data=True):
                        # if there is still one adjacent edge from curr_module or with index e==curr_module , update LP
                        if data_['m'] == curr_module or data_['e'] == curr_module: 
                            still_connected = True
                            lp = calc_lp(e,m,curr_module,edge2Patients,nOnesPerPatientInModules,moduleSizes,
                                         moduleOneFreqs,p0,match_score,mismatch_score,bK_1,
                                         alpha=alpha,beta_K=beta_K)
                            m_ndx = data['modules'].index(curr_module)
                            data['log_p'][m_ndx] = lp
                            
                            break

                    # if not connected, remove curr_m from the list
                    if not still_connected:
                        m_ndx = data['modules'].index(curr_module)
                        del data['modules'][m_ndx]
                        del data['log_p'][m_ndx]


def get_consensus_modules(edge2module_history, network, edge2Patients, edge2Module,
                          nOnesPerPatientInModules,moduleSizes, moduleOneFreqs, p0, match_score,mismatch_score,
                          bK_1,alpha=1.0,beta_K=1.0):
    consensus_edge2module = []
    labels = np.asarray(edge2module_history)
    
    # identify modules which edges ocsilate 
    edges = network.edges()
    for i in range(0,len(edges)):
        unique, counts = np.unique(labels[:,i], return_counts=True)
        if len(unique) >1:
            counts = np.array(counts)
            new_ndx = unique[np.argmax(counts)]
            if float(max(counts))/labels.shape[0] < 0.5: 
                print("Warning: less than 50% of time in the most frequent module\n\tedge:",i,
                      "counts:",counts,"\n\tlabels:" , ",".join(map(str,unique)) ,file= sys.stdout)
            consensus_edge2module.append(new_ndx)
        else:
            consensus_edge2module.append(unique[0])
            
    # construct consensus edge-to-module membership
    i =0 
    changed_edges = 0
    for i in range(0,len(consensus_edge2module)):
        curr_module = edge2Module[i]
        new_module = consensus_edge2module[i]
        if curr_module != new_module:
            changed_edges += 1
            n1, n2 = edges[i]
            apply_changes(network,n1,n2,i,curr_module,new_module,
                          edge2Patients, edge2Module, nOnesPerPatientInModules,moduleSizes,
                          moduleOneFreqs,p0,match_score,mismatch_score,bK_1,
                          alpha=alpha,beta_K=beta_K,no_LP_update = True)
            
    print(changed_edges, "edges changed their module membership after taking consensus.")
    return consensus_edge2module, network, edge2Patients, nOnesPerPatientInModules, moduleSizes, moduleOneFreqs

################################## 3. Post-processing ##########################################

def get_genes(mid,edge2module,edges):
    ndx = [i for i, j in enumerate(edge2module) if j == mid]
    genes = []
    for edge in [edges[i] for i in ndx]:
        genes.append(edge[0])
        genes.append(edge[1])
    genes = list(set(genes))
    return genes

def calc_bic_SNR(genes, samples, exprs, N, exprs_sums,exprs_sq_sums):
    bic = exprs[genes,:][:,samples]
    bic_sums = bic.sum(axis=1)
    bic_sq_sums = np.square(bic).sum(axis=1)

    bg_counts = N - len(samples)
    bg_sums = exprs_sums[genes]-bic_sums
    bg_sq_sums = exprs_sq_sums[genes]-bic_sq_sums
    
    bic_mean, bic_std = calc_mean_std_by_powers((len(samples),bic_sums,bic_sq_sums))
    bg_mean, bg_std = calc_mean_std_by_powers((bg_counts,bg_sums,bg_sq_sums))
    
    return  np.mean(abs(bic_mean - bg_mean)/ (bic_std + bg_std))

def bicluster_avg_SNR(exprs,genes=[],samples=[]):
    mat = exprs.loc[genes, :]
    all_samples = set(mat.columns.values)
    bic= mat.loc[:, samples]
    out = mat.loc[:, all_samples.difference(samples)]
    SNR = (bic.mean(axis=1) - out.mean(axis=1))/(bic.std(axis=1)+out.std(axis=1))
    return np.mean(abs(SNR))

def bicluster_corrs(exprs,genes=[],samples=[]):
    if len(samples) > 0 :
        mat = exprs.loc[genes,samples]
    else:
        mat = exprs.loc[genes, :]
    corrs  = []
    for g1,g2 in itertools.combinations(genes,2):
        corr = np.corrcoef(mat.loc[g1,:].values, mat.loc[g2,:].values)[0][1]
        corrs.append(corr)
    return round(np.average(corrs),2), round(np.min(corrs),2), round(np.max(corrs),2)

def identify_opt_sample_set(genes,exprs,exprs_data,direction="UP",min_n_samples=8):
    N, exprs_sums, exprs_sq_sums = exprs_data
    e = exprs[genes,:]
    
    labels = KMeans(n_clusters=2, random_state=0).fit(e.T).labels_
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1),len(ndx0))< min_n_samples:
        return {"avgSNR":-1}
    if np.mean(e[:,ndx1].mean()) > np.mean(e[:,ndx0].mean()):
        if direction=="UP":samples = ndx1
        else: samples = ndx0
    else:
        if direction=="UP":samples = ndx0
        else: samples = ndx1
    avgSNR = calc_bic_SNR(genes, samples, exprs, N, exprs_sums, exprs_sq_sums)

    if len(samples)<N*0.5*1.1 and len(samples)>=min_n_samples: # allow bicluster to be a little bit bigger than N/2
        bic = {"genes":set(genes),"n_genes":len(genes),
               "samples":set(samples),"n_samples":len(samples),
               "avgSNR":avgSNR,"direction":direction}
        return bic
    else:
        return {"avgSNR":-1}

def identify_opt_sample_set_for_discordant(genes,exprs,exprs_data,min_n_samples=8):
    N, exprs_sums, exprs_sq_sums = exprs_data
    e = exprs[genes,:]
    
    labels = KMeans(n_clusters=2, random_state=0).fit(e.T).labels_
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1),len(ndx0))< min_n_samples:
        return {"avgSNR":-1}
    
    # biclsuter is a smaller group, direction = "MIXED"
    if len(ndx0) <= len(ndx1):
        samples = ndx0
    else:
        samples = ndx1
        
    avgSNR = calc_bic_SNR(genes, samples, exprs, N, exprs_sums, exprs_sq_sums)

    bic = {"genes":set(genes),"n_genes":len(genes),
               "samples":set(samples),"n_samples":len(samples),
               "avgSNR":avgSNR,"direction":"MIXED"}
    return bic

#### Merging  ### 

def calc_J(bic,bic2,all_samples):
    s1 = bic["samples"]
    s2 = bic2["samples"]
    if bic["direction"] != bic2["direction"]:
        s2 = all_samples.difference(s2)
    sh_samples = s1.intersection(s2)
    u_samples = s1.union(s2)
    J=1.0*len(sh_samples)/len(u_samples)
    return J

def calc_overlap_pval_J(bic,bic2,all_samples):
    s1 = bic["samples"]
    s2 = bic2["samples"]
    if bic["direction"] != bic2["direction"]:
        s2 = all_samples.difference(s2)
    s1_s2 = len(s1.intersection(s2))
    s1_only = len(s1.difference(s2))
    s2_only = len(s2.difference(s1))
    p_val = pvalue(s1_s2,s1_only,s2_only,len(all_samples)-s1_s2-s1_only-s2_only).right_tail
    J=1.0*s1_s2/(s1_s2+s1_only+s2_only )
    return p_val, J


def merge_biclusters(biclusters, exprs, exprs_data, min_n_samples=5,
                     verbose = True, min_SNR=0.5,J_threshold=0.25,pval_threshold=0.05,
                     merge_discordant=False):
    print("merge_discordant",merge_discordant)
    t0 = time.time()
    
    n_bics = len(biclusters)
    bics = dict(zip(range(0,len(biclusters)), biclusters))

    all_samples = set(range(0,exprs.shape[1]))
    candidates = {}
    for i in bics.keys():
        bic = bics[i]
        for j in bics.keys():
            if j>i:
                bic2 = bics[j]
                p_val, J = calc_overlap_pval_J(bic,bic2,all_samples)
                
                p_val=p_val*n_bics
                if p_val<pval_threshold:# and J>J_threshold: 
                    candidates[(i,j)] = p_val,J
                     
    nothing_to_merge = False
    while not nothing_to_merge:       
        # take min p-value pair
        opt_pval, opt_J = pval_threshold, 0.0
        opt_pair = -1, -1
        
        for pair in candidates.keys():
            p_val,J = candidates[pair]
            if p_val < opt_pval:
                opt_pval, opt_J = p_val,J
                opt_pair  = pair
            elif p_val == opt_pval: # take max(J) in case of ties
                if J > opt_J:
                    opt_pval, opt_J = p_val,J
                    opt_pair  = pair
        
        if opt_pair[0] == -1:
            nothing_to_merge = True
        else:
            opt_i, opt_j = opt_pair
            if verbose:
                print("\t\ttry merging %s:%sx%s  (%s,%s) + %s:%sx%s  (%s,%s)"%(bics[opt_i]["id"],bics[opt_i]["n_genes"],
                                                                           bics[opt_i]["n_samples"],
                                                                           round(bics[opt_i]["avgSNR"],2),
                                                                           bics[opt_i]["direction"],
                                                                           bics[opt_j]["id"],bics[opt_j]["n_genes"],
                                                                           bics[opt_j]["n_samples"], 
                                                                           round(bics[opt_j]["avgSNR"],2),
                                                                           bics[opt_j]["direction"]))
            
            # try creating a new bicsluter from bic and bic2
            genes = list(bics[opt_i]["genes"] | bics[opt_j]["genes"])
            
            
            if merge_discordant:
                if bics[opt_i]["direction"]!=bics[opt_j]["direction"]:
                    new_bic = identify_opt_sample_set_for_discordant(genes,exprs,exprs_data,min_n_samples=min_n_samples)
                else:
                    new_bic = identify_opt_sample_set(genes,exprs,exprs_data, 
                                                  direction=bics[opt_i]["direction"],
                                                  min_n_samples=min_n_samples)
            else:
                new_bic = identify_opt_sample_set(genes,exprs,exprs_data, 
                                                  direction=bics[opt_i]["direction"],
                                                  min_n_samples=min_n_samples)
                if new_bic["avgSNR"]==-1 and bics[opt_i]["direction"]!=bics[opt_j]["direction"]:
                    new_bic = identify_opt_sample_set(genes, exprs, exprs_data,
                                                          direction=bics[opt_j]["direction"],
                                                          min_n_samples=min_n_samples)

            avgSNR = new_bic["avgSNR"]
            if avgSNR >= min_SNR:
                # place new_bic to ith bic
                new_bic["id"] = bics[opt_i]["id"]
                if verbose:
                    substitution = (bics[opt_i]["id"], len(bics[opt_i]["genes"]),len(bics[opt_i]["samples"]),
                                    round(bics[opt_i]["avgSNR"],2),bics[opt_i]["direction"],
                                    bics[opt_j]["id"], len(bics[opt_j]["genes"]), 
                                    len(bics[opt_j]["samples"]),
                                    round(bics[opt_j]["avgSNR"],2),bics[opt_j]["direction"],
                                    round(new_bic["avgSNR"],2),len(new_bic["genes"]),
                                    len(new_bic["samples"]))
                    print("\tMerge biclusters %s:%sx%s (%s,%s) and %s:%sx%s  (%s,%s) --> %s SNR and %sx%s"%substitution)
                new_bic["n_genes"] = len(new_bic["genes"])
                new_bic["n_samples"] = len(new_bic["samples"])
                
                bics[opt_i] = new_bic
                # deleted data for ith and jth biclusters
                for i,j in candidates.keys():
                    if i == opt_j or j == opt_j:
                        del candidates[(i,j)]
                # remove j-th bics jth column and index
                del bics[opt_j]
                n_bics = len(bics)
                for j in bics.keys():
                    if j!=opt_i:
                        J = calc_J(new_bic,bics[j],all_samples)
                        p_val,J = calc_overlap_pval_J(new_bic,bics[j],all_samples)
                        p_val = p_val*n_bics
                        if p_val<pval_threshold: # and J>J_threshold:
                            if opt_i <j:
                                candidates[(opt_i,j)] = p_val,J
                            else:
                                candidates[(j,opt_i)] = p_val,J
            else:
                # set J for this pair to 0
                if verbose:
                    print("\t\tSNR=", round(avgSNR,2),"<",round(min_SNR,2),"--> no merge")
                candidates[(opt_i,opt_j)] = pval_threshold,0.0
     
    if verbose:
        print("time:\tMerging finished in:",round(time.time()-t0,2))
     
    return bics.values()