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
def define_SNR_threshold(snr_file, exprs,network, q, min_n_samples=5, random_sample_size= 1000):
    from sklearn.cluster import KMeans

    if not os.path.exists(snr_file):
    
        avgSNR = []
        t0 = time.time()
        edges = network.edges()
        lens = []
        for i in range(0, random_sample_size):
            g1,g2 = random.choice(edges)
            e = exprs.loc[[g1,g2],:]
            labels = KMeans(n_clusters=2, random_state=0,n_init=1,max_iter=100).fit(e.T).labels_
            ndx0 = np.where(labels == 0)[0]
            ndx1 = np.where(labels == 1)[0]
            if min(len(ndx1),len(ndx0))< min_n_samples:
                avgSNR.append(0)
            e1 = e.iloc[:,ndx1]
            e0 = e.iloc[:,ndx0]
            SNR0 = np.mean(e0.mean(axis=1)-e1.mean(axis=1)/(e0.std(axis=1)+e1.std(axis=1)))
            if np.isnan(SNR0):
                SNR0 = 0 
            avgSNR.append(abs(SNR0))

        min_SNR = np.quantile(avgSNR,q=1-q)
        f = open(snr_file,"w")
        print(min_SNR,file=f)
    else:
        f = open(snr_file,"r")
        min_SNR = float(f.readlines()[0])
    f.close()
    return (min_SNR)


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

    
def partialRRHO(gene1,gene2,subnet,rhho_thrs,min_SNR=0.5,min_n_samples=8,fixed_step=10, verbose = False):
    '''Searches for thresholds corresponding to optimal patient overlap. 
    The overlap is optimal if
    \n1). it passes significance threshold in hypergeometric test;
    \n2). resulting set of samples inuces a bicluster with SNR higher than 'min_SNR' threshold;
    \n3). its size is maximal.
    \n
    \nReturns group of samples with gene expressions above selected thresholds.
    \ngene1, gene2 - genes to compare,
    \nexprs_profile_dict - a dictionary with gene expression profiles
    '''
    t_0 = time.time()
    e1 = subnet.node[gene1]["exprs"]
    e2 = subnet.node[gene2]["exprs"]
    rlist1 = e1.index.values
    rlist2 = e2.index.values

    # pick point with maximum overlap with p-value < thr
    p_val = 1
    t_u0 = len(rlist1)/2
    t_w0 = len(rlist2)/2
    t_u, t_w = t_u0, t_w0
    i=0
    optimal_thresholds = (-1,t_u0)
    optimal_patient_set  = set()
    stop_condition = False  # found significant overlap or thresolds reached ends of both lists 
    while not stop_condition :
        # generate combinations of thresholds cutting the same number of patients 
        for elem in itertools.product(range(0,i+1), repeat= 2):
            if elem[0]+elem[1] == i:
                # move thresholds
                t_u = t_u0-fixed_step*elem[0]
                t_w = t_w0-fixed_step*elem[1]
                if t_u > 0 and t_w >0:
                    #p_val, enrichment, samples = hg_test(rlist1,rlist2,t_u,t_w,return_items=True)
                    samples = set(rlist1[:t_u]).intersection(set(rlist2[:t_w]))
                    if len(samples) > rhho_thrs[t_u][t_w]:
                        SNR = []
                        for e in [e1,e2]:
                            e_in = e[e.index.isin(samples)].values
                            e_out = e[~e.index.isin(samples)].values
                            SNR.append(abs(e_in.mean()-e_out.mean())/(e_in.std()+e_out.std()))
                        SNR = (SNR[0]+SNR[1])/2
                        if SNR >= min_SNR:
                            stop_condition = True
                            if len(samples) > len(optimal_patient_set):
                                optimal_thresholds = (t_u, t_w)
                                optimal_patient_set = samples
                            elif len(samples) == len(optimal_patient_set):
                                if abs(t_u-t_w) < abs(optimal_thresholds[0] - optimal_thresholds[1]):
                                    optimal_thresholds = (t_u, t_w)
                                    optimal_patient_set = samples
                            else:
                                pass
                    if len(samples) <= min_n_samples:
                        stop_condition = True 
                else:
                    stop_condition=True
                    break
                                
        i+=1
    if verbose:
        if len(optimal_patient_set) >0 :
            print("runtime:",round(time.time()-t_0,5),"s","Best ovelap for",(gene1,gene2),"occurs at thresholds",optimal_thresholds,"and contains",len(optimal_patient_set),"samples",file = sys.stderr)
        else:
            print("runtime:",round(time.time()-t_0,5),"s","No significant overlap for",(gene1,gene2),file = sys.stderr)
    #print("partialRRHO() runtime:",round(time.time()-t_0,5),"s",file=sys.stderr)        
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
    # exclude nodes without expressoin from the network
    nodes = set(subnet.nodes())
    genes_with_expression = set(exprs.index.values)
    nodes_with_exprs = nodes.intersection(genes_with_expression)
    if len(nodes_with_exprs) != len(nodes):
        if verbose:
            print("Nodes in subnetwork:",len(nodes),"of them with expression:",len(nodes_with_exprs),file=sys.stdout)
        subnet = subnet.subgraph(list(nodes_with_exprs)).copy()
    # keep only genes corresponding nodes in the expression matrix 
    e = exprs.loc[nodes_with_exprs,:].T
    if verbose:
        print("Samples x Genes in expression matrix",e.shape, file=sys.stdout)
    
    # assign expression profiles to every node  
    node2exprs = {}
    # store only sorted sample names
    for gene in e.columns.values:
        node2exprs[gene] = e[gene].sort_values(ascending=ascending)
    
    ## set node attributes
    nx.set_node_attributes(subnet, 'exprs', node2exprs)
    print("expression_profiles2nodes()\truntime:",round(time.time()-t_0,5),"s",file=sys.stdout)
    return subnet

def assign_patients2edges(subnet,min_SNR=0.5,min_n_samples=8, fixed_step=10,rrho_thrs = False, verbose=True):
    t0 = time.time()
    # mark all edges as not masked
    # assign samples to every edge: 
    edge2pats = {}
    i = 0
    runtimes = []
    for edge in subnet.edges():
        n1, n2 = edge
        t_0 = time.time()
        up = partialRRHO(n1,n2,subnet,rrho_thrs,min_SNR=min_SNR,min_n_samples=min_n_samples,
                            fixed_step=fixed_step,verbose=False)
        runtimes.append(time.time() - t_0)
        i+=1
        if i%1000 == 0 and verbose:
            print("\t",i,"edges processed. Average runtime per edge:",np.mean(runtimes), file=sys.stdout )
        #if n1 in seeds2 and n2 in seeds2:
        #    print(edge, len(up), len(down))
        #edge2pats[edge] = up
        subnet[n1][n2]["samples"] = up
        subnet[n1][n2]["masked"] = False
    print("assign_patients2edges()\truntime, s:", round(time.time() - t0,4),"for subnetwork of",len(subnet.nodes()),"nodes and",
          len(subnet.edges()),"edges.", file = sys.stdout)
    #nx.set_edge_attributes(subnet, 'samples', edge2pats)
    return subnet

#### get rid of empty edges after RRHO ####
def mask_empty_edges(network,min_n_samples=10,remove = False, verbose = True):
    t_0 = time.time()
    n_pats = []
    n_masked = 0
    n_retained = 0
    for edge in network.edges():
        n1,n2 = edge
        pats = len(network[n1][n2]["samples"])
        n_pats.append(pats)
        # mask edges with not enough samples 
        if pats < min_n_samples:
            if remove:
                network.remove_edge(n1,n2)
            else:
            # just mask
                network[n1][n2]["masked"] = True
            n_masked += 1
        else:
            n_retained += 1
    if verbose:
        print("Of total %s edges %s retained;\n%s edges containing less than %s samples were masked/removed" % 
          (n_masked+n_retained, n_retained, n_masked, min_n_samples), file=sys.stdout)
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stdout)
    return network
#################################### Gibbs Sampling #######################################################

def calc_lp(edge,self_module,module,edge2Patients,
            nOnesPerPatientInModules,moduleSizes,
            moduleOneFreqs,p0,match_score,mismatch_score,bK_1,log_func=np.log,
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
    ones_matching_term = np.inner(log_func(oneRatios),edge_vector)
    # zero-matching
    zeroRatios = (m_size-n_ones_per_pat+alpha/2)/(m_size+alpha)
    zeros_matching_term = np.inner(log_func(zeroRatios),(1-edge_vector))

    return ones_matching_term+zeros_matching_term + beta_term

def set_initial_conditions(network,exprs, p0, match_score, mismatch_score, bK_1, N,
                           log_func=np.log, alpha = 1.0,
                           beta_K = 1.0,verbose = True):
    t_0 = time.time()
    
    # 1. the number of edges inside each component, initially 1 for each component
    moduleSizes=np.ones(len(network.edges()),dtype=np.int)
    
    # 2. a binary (int) matrix of size n by m that indicates the samples on the edges
    edge2Patients = []
    all_pats = list(exprs.columns.values)
    for edge in network.edges():
        n1,n2 = edge
        pats_in_module = network[n1][n2]["samples"]
        x = np.zeros(len(all_pats), dtype=np.int)
        i = 0
        for p in all_pats:
            if p in pats_in_module:
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
        del data['masked']
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
                                 bK_1,log_func=np.log,
                                 alpha=alpha,beta_K=beta_K)
                    data['log_p'].append(lp)
                    data['modules'].append(m2)
        if verbose and e%1000 == 0:
            print(e,"\tedges processed",round(time.time()- t_1,1) , "s runtime",file=sys.stdout)
            t_1 = time.time()
    print("\tSet initial LPs in",round(time.time()- t_0,1) , "s", file = sys.stdout)
    
    print("time: Initial state created in",round(time.time()- t_0,1) , "s runtime", file = sys.stdout)
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
def sampling(network, exprs,edge2Module, edge2Patients,nOnesPerPatientInModules,moduleSizes,
             moduleOneFreqs, p0, match_score, mismatch_score, bK_1,log_func=np.log, alpha = 0.1, beta_K = 1.0,
             max_n_steps=100,n_steps_averaged = 20,n_points_fit = 10,tol = 0.1,
             n_steps_for_convergence = 5,verbose=True, edge_ordering ="corr"):
    if edge_ordering =="corr":
        ### define edge order depending on gene expression corelations:
        t0= time.time()
        from scipy.stats.stats import pearsonr
        edge_corrs = []
        for g1,g2 in network.edges():
            r,pv = pearsonr(exprs.loc[g1,:].values,exprs.loc[g2,:].values)
            edge_corrs.append(-r)
        edge_order = np.argsort(edge_corrs)
        print("Top correlations:",edge_corrs[edge_order[0]],edge_corrs[edge_order[1]],edge_corrs[edge_order[2]])
        edge_corrs = [] # do not store
        print("Correlations computed in %s s" % round(time.time()-t0,2))
    else:
        # shuffle edges
        edge_order = range(0, edge2Patients.shape[0])
    t_ =  time.time()
    edge2Module_history = [copy.copy(edge2Module)]
    #n_edge_skip_history = [0]
    #max_edges_oscillating = int(max_frac_edges_oscillating*len(network.edges()))
    is_converged = False
    network_edges = network.edges(data=True)
    for step in range(1,max_n_steps):
        if verbose:
            print("step",step,file = sys.stdout)
        not_changed_edges = 0
        t_0 = time.time()
        t_1=t_0
        i = 1
        if edge_ordering == "shuffle":
            random.shuffle(edge_order)
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
                              moduleOneFreqs, p0,match_score,mismatch_score, bK_1,log_func=log_func,
                              alpha=alpha,beta_K=beta_K)
                
            else:
                not_changed_edges +=1#
            i+=1
            if i%10000 == 1:
                if verbose:
                    print(i,"edges processed", "\t",
                          1.0*not_changed_edges/len(edge_order),
                          "%edges not changed;",
                          round(time.time()- t_1,1) , "s runtime",file=sys.stdout)
                not_changed_edges=0
                t_1 = time.time()
        if verbose:
            print("\tstep",step,1.0*not_changed_edges/len(edge_order), 
                  "- % edges not changed; runtime",round(time.time()- t_0,1) , "s", file = sys.stdout)
        edge2Module_history.append(copy.copy(edge2Module))
        #n_edge_skip_history.append(len(network.edges())-not_changed_edges)
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
                  moduleOneFreqs,p0,match_score,mismatch_score,bK_1,log_func=np.log,
                  alpha=1.0,beta_K=1.0,no_LP_update = False):
    '''Moves the edge from current module to the new one
    and updates network, nOnesPerPatientInModules and moduleSizes respectively.'''
    # update the edge module membership
    #print("edge",edge_ndx,n1+'-'+n2,"module:",curr_module,"-->",new_module)
    network[n1][n2]['m'] = new_module
    edge2Module[edge_ndx] = new_module
    # for this edge no probabilities change
    
    # reduce curr_module size and nOnesPerPatientInModules
    edge_vector = edge2Patients[edge_ndx,]
    nOnesPerPatientInModules[curr_module,] = nOnesPerPatientInModules[curr_module,] - edge_vector
    moduleSizes[curr_module,]-=1
    #moduleOneFreqs[curr_module] =  sum([1.0 for x in nOnesPerPatientInModules[curr_module,] if x >= float(moduleSizes[curr_module,])/2])/N
    
    # increase new_module
    nOnesPerPatientInModules[new_module,] = nOnesPerPatientInModules[new_module,] + edge_vector
    moduleSizes[new_module,]+=1
    #moduleOneFreqs[new_module] = sum([1.0 for x in nOnesPerPatientInModules[new_module,] if x >= float(moduleSizes[new_module,])/2])/N
    
    # update LP for all edges contacting curr and new modules, except skipping edge
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
                                 moduleOneFreqs,p0,match_score,mismatch_score,bK_1,log_func=log_func,
                                 alpha=alpha,beta_K=beta_K)
                    # find index of targe module or add it if did not exist
                    if new_module in data['modules']:
                        m_ndx = data['modules'].index(new_module)
                        data['log_p'][m_ndx] = lp
                        #print("\tupdated",n+'-'+n3,"LP for new_module",new_module)
                    else: # if it is a novel connection to a new_module, append it to the end of a list
                        data['modules'].append(new_module)
                        data['log_p'].append(lp)
                        #print("\tadded to",n+'-'+n3,"new_module",new_module)
                    #### update LP for curr_module
                    # check if an edge is still connected with curr_module
                    still_connected = False
                    # iterate all edges adjacent to affected
                    for n1_,n2_,data_ in network.edges([n,n3],data=True):
                        # if there is still one adjacent edge from curr_module or with index e==curr_module , update LP
                        if data_['m'] == curr_module or data_['e'] == curr_module: 
                            still_connected = True
                            lp = calc_lp(e,m,curr_module,edge2Patients,nOnesPerPatientInModules,moduleSizes,
                                         moduleOneFreqs,p0,match_score,mismatch_score,bK_1,log_func=log_func,
                                         alpha=alpha,beta_K=beta_K)
                            m_ndx = data['modules'].index(curr_module)
                            data['log_p'][m_ndx] = lp
                            #print("\tupdated",n+'-'+n3,"LP for curr_module",new_module)
                            break

                    # if not connected, remove curr_m from the list
                    if not still_connected:
                        m_ndx = data['modules'].index(curr_module)
                        del data['modules'][m_ndx]
                        del data['log_p'][m_ndx]
                        #print("change edge.m",e,n,n3,"belonging to",m,";","for",curr_module,"-->",new_module)
                        #print("\tremoved from",n+'-'+n3,"cur_module",new_module)


### get consensus module membership
def get_consensus_modules(edge2module_history, network, edge2Patients, edge2Module,
                          nOnesPerPatientInModules,moduleSizes, moduleOneFreqs, p0, match_score,mismatch_score,
                          bK_1,log_func=np.log,
                          alpha=1.0,beta_K=1.0):
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
                # TODO: put such edges to any empty neighbouring module
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
                          moduleOneFreqs,p0,match_score,mismatch_score,bK_1,log_func=log_func,
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

def identify_opt_sample_set(genes, exprs,direction="UP",min_n_samples=8):
    e = exprs.loc[genes,:]
    N = exprs.shape[1]
    labels = KMeans(n_clusters=2, random_state=0).fit(e.T).labels_
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1),len(ndx0))< min_n_samples:
        return {"avgSNR":-1}
    if np.mean(e.iloc[:,ndx1].mean()) > np.mean(e.iloc[:,ndx0].mean()):
        if direction=="UP":ndx = ndx1
        else: ndx = ndx0
    else:
        if direction=="UP":ndx = ndx0
        else: ndx = ndx1
    samples = e.columns[ndx]
    avgSNR = bicluster_avg_SNR(exprs,genes=genes,samples=samples)

    if len(samples)<N*0.5*1.1 and len(samples)>=min_n_samples: # allow bicluster to be a little bit bigger
        bic = {"genes":set(genes), "samples":set(samples), "avgSNR":avgSNR,"direction":direction}
        return bic
    else:
        return {"avgSNR":-1}


#### Merging  ### 
def find_dfmax(df,maxJ=0.75):
    df = df.dropna(how="all",axis=1)
    df.dropna(how="all",inplace=True,axis=0)
    rowmax = df.idxmax(skipna=True,axis=1)
    maxi, maxj, maxJ = None,None,maxJ
    for i in rowmax.index.values:
        j = rowmax[i]
        J = df.loc[i,j]

        if J>maxJ:
            maxJ = J
            maxi, maxj = i,j
    return maxi, maxj, maxJ

def calc_J(bic,bic2,all_samples):
    s1 = bic["samples"]
    s2 = bic2["samples"]
    if bic["direction"] != bic2["direction"]:
        s2 = all_samples.difference(s2)
    sh_samples = s1.intersection(s2)
    u_samples = s1.union(s2)
    J=1.0*len(sh_samples)/len(u_samples)
    return J

def calc_overlap_pval(bic,bic2,all_samples):
    s1 = bic["samples"]
    s2 = bic2["samples"]
    if bic["direction"] != bic2["direction"]:
        s2 = all_samples.difference(s2)
    s1_s2 = len(s1.intersection(s2))
    s1_only = len(s1.difference(s2))
    s2_only = len(s2.difference(s1))
    p_val = pvalue(s1_s2,s1_only,s2_only,len(all_samples)-s1_s2-s1_only-s2_only).right_tail
    return p_val

def merge_biclusters(biclusters, exprs, min_n_samples=8,
                     verbose = True, min_SNR=0.5,
                     max_SNR_decrease=0.1, J_threshold=0.5,pval_threshold=0.05):
    t0 = time.time()
    
    bics = dict(zip(range(0,len(biclusters)), biclusters))

    all_samples = set(exprs.columns.values)
    candidates = {}
    for i in bics.keys():
        bic = bics[i]
        for j in bics.keys():
            if j>i:
                bic2 = bics[j]
                J = calc_J(bic,bic2,all_samples) 
                p_val = calc_overlap_pval(bic,bic2,all_samples)
                if J>J_threshold and p_val<pval_threshold:
                    candidates[(i,j)] = J
                     
    nothing_to_merge = False
    
    while not nothing_to_merge and len(candidates.values())>0:       
        # take max J pair
        max_J = max(candidates.values())
        if max_J<J_threshold:
            nothing_to_merge = True
        else:
            maxi, maxj = candidates.keys()[candidates.values().index(max_J)]
            print("\t\ttry merging %s:%sx%s  (%s,%s) + %s:%sx%s  (%s,%s)"%(bics[maxi]["id"],bics[maxi]["n_genes"],
                                                                           bics[maxi]["n_samples"],
                                                                           round(bics[maxi]["avgSNR"],2),
                                                                           bics[maxi]["direction"],
                                                                           bics[maxj]["id"],bics[maxj]["n_genes"],
                                                                           bics[maxj]["n_samples"], 
                                                                           round(bics[maxj]["avgSNR"],2),
                                                                           bics[maxj]["direction"]))
            
            # try creating a new bicsluter from bic and bic2
            genes = bics[maxi]["genes"] | bics[maxj]["genes"] 
            new_bic = identify_opt_sample_set(genes, exprs,
                                              direction=bics[maxi]["direction"],
                                              min_n_samples=min_n_samples)
            if new_bic["avgSNR"]==-1 and bics[maxi]["direction"]!=bics[maxj]["direction"]:
                new_bic = identify_opt_sample_set(genes, exprs,direction=bics[maxj]["direction"],
                                                  min_n_samples=min_n_samples)

            avgSNR = new_bic["avgSNR"]
            if avgSNR >= min_SNR:
                # place new_bic to ith bic
                new_bic["id"] = bics[maxi]["id"]
                substitution = (bics[maxi]["id"], len(bics[maxi]["genes"]),len(bics[maxi]["samples"]),
                                    round(bics[maxi]["avgSNR"],2),bics[maxi]["direction"],
                                    bics[maxj]["id"], len(bics[maxj]["genes"]), 
                                    len(bics[maxj]["samples"]),
                                    round(bics[maxj]["avgSNR"],2),bics[maxj]["direction"],
                                    round(new_bic["avgSNR"],2),len(new_bic["genes"]),
                                    len(new_bic["samples"]))
                print("\tMerge biclusters %s:%sx%s (%s,%s) and %s:%sx%s  (%s,%s) --> %s SNR and %sx%s"%substitution)
                new_bic["n_genes"] = len(new_bic["genes"])
                new_bic["n_samples"] = len(new_bic["samples"])
                
                bics[maxi] = new_bic
                # deleted J data for ith and jth biclusters
                for i,j in candidates.keys():
                    if i == maxj or j == maxj:
                        del candidates[(i,j)]
                # remove j-th bics jth column and index
                del bics[maxj]
                for j in bics.keys():
                    if j!=maxi:
                        J = calc_J(new_bic,bics[j],all_samples)
                        p_val = calc_overlap_pval(new_bic,bics[j],all_samples)
                        if J>J_threshold and p_val<pval_threshold:
                            if maxi <j:
                                candidates[(maxi,j)] = J
                            else:
                                candidates[(j,maxi)] = J
            else:
                # set J for this pair to 0
                print("\t\tSNR=", round(avgSNR,2),"<",round(min_SNR,2),"--> no merge")
                candidates[(maxi,maxj)] = 0
     
    if verbose:
        print("time:\tMerging finished in:",round(time.time()-t0,2))
     
    return bics
