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



import networkx as nx


import matplotlib
import matplotlib.pyplot as plt 

###################################################### RRHO ####################################################################

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

def partialRRHO(gene1,gene2,subnet,rhho_thrs,fixed_step=10, verbose = False):
    '''Searches for thresholds corresponding to optimal patient overlap. 
    The overlap is optimal if 1). it passes significance threshold in hypergeometric test 2). its size is maximal 
    \nReturns group of patients with gene expressions above selected thresholds.
    \ngene1, gene2 - genes to compare,
    \nexprs_profile_dict - a dictionary with gene expression profiles
    '''
    t_0 = time.time()
    rlist1 = subnet.node[gene1]["exprs"].index.values
    rlist2 = subnet.node[gene2]["exprs"].index.values

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
                    #p_val, enrichment, patients = hg_test(rlist1,rlist2,t_u,t_w,return_items=True)
                    patients = set(rlist1[:t_u]).intersection(set(rlist2[:t_w]))
                    if len(patients) > rhho_thrs[t_u][t_w]:
                        stop_condition = True
                        if len(patients) > len(optimal_patient_set):
                            optimal_thresholds = (t_u, t_w)
                            optimal_patient_set = patients
                        elif len(patients) == len(optimal_patient_set):
                            if abs(t_u-t_w) < abs(optimal_thresholds[0] - optimal_thresholds[1]):
                                optimal_thresholds = (t_u, t_w)
                                optimal_patient_set = patients
                        else:
                            pass
                else:
                    stop_condition=True
                    break
                                
        i+=1
    if verbose:
        if len(optimal_patient_set) >0 :
            print("runtime:",round(time.time()-t_0,5),"s","Best ovelap for",(gene1,gene2),"occurs at thresholds",optimal_thresholds,"and contains",len(optimal_patient_set),"patients",file = sys.stderr)
        else:
            print("runtime:",round(time.time()-t_0,5),"s","No significant overlap for",(gene1,gene2),file = sys.stderr)
    #print("partialRRHO() runtime:",round(time.time()-t_0,5),"s",file=sys.stderr)        
    return optimal_patient_set

#### Assign sets of patients on edges ####
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

def assign_patients2edges(subnet, method="top_half",fixed_step=10,rrho_thrs = False, verbose=True):
    t0 = time.time()
    # mark all edges as not masked
    # assign patients to every edge: 
    edge2pats = {}
    i = 0
    runtimes = []
    for edge in subnet.edges():
        n1, n2 = edge
        t_0 = time.time()
        if method=="top_half":
            # just takes overlap of top_n in both lists 
            p1 = subnet.node[n1]["exprs"].index.values
            p2 = subnet.node[n2]["exprs"].index.values
            up = top_n(p1,p2,n=len(p1)/2)
        elif method=="RRHO":
            #if not type(rrho_thresholds) == pd.core.frame.DataFrame:
            #    print("Pre-compute thresholds for RRHO",file = sys.stderr)
            up = partialRRHO(n1,n2,subnet,rrho_thrs,fixed_step=fixed_step,verbose=False)
        else:
            if verbose : 
                print("'method' must be one of 'top_half' or 'RRHO'.",file = sys.stdout)
        runtimes.append(time.time() - t_0)
        i+=1
        if i%1000 == 0 and verbose:
            print("\t",i,"edges processed. Average runtime per edge:",np.mean(runtimes), file=sys.stdout )
        #if n1 in seeds2 and n2 in seeds2:
        #    print(edge, len(up), len(down))
        #edge2pats[edge] = up
        subnet[n1][n2]["patients"] = up
        subnet[n1][n2]["masked"] = False
    print("assign_patients2edges()\truntime, s:", round(time.time() - t0,4),"for subnetwork of",len(subnet.nodes()),"nodes and",
          len(subnet.edges()),"edges.", file = sys.stdout)
    #nx.set_edge_attributes(subnet, 'patients', edge2pats)
    return subnet


#### get rid of empty edges after RRHO ####
def mask_empty_edges(network,min_n_samples=10,remove = False, verbose = True):
    t_0 = time.time()
    n_pats = []
    n_masked = 0
    n_retained = 0
    for edge in network.edges():
        n1,n2 = edge
        pats = len(network[n1][n2]["patients"])
        n_pats.append(pats)
        # mask edges with not enough patients 
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


def calc_norm_coef(p,q,N,match_score,mismatch_score):
    p_match = (2*p*q+1-p -q)
    return (p_match*match_score+(1-p_match)*mismatch_score)*N

def calc_lp(edge,self_module,module,edge2Patients,nOnesPerPatientInModules,moduleSizes,
            edgeOneFreqs, moduleOneFreqs,p0,
            alpha=1.0,beta_K=1.0):
    
    m_size = moduleSizes[module]
    edge_vector = edge2Patients[edge,]
    #p = edgeOneFreqs[edge]
    if m_size == 0:
        return p0#-calc_norm_coef(p,p)
    

    n_ones_per_pat = nOnesPerPatientInModules[module,]
    
    if self_module == module: # remove the edge from the module, if it belongs to a module
        if m_size == 1:
            return p0 #- calc_norm_coef(p,p)
        m_size -=1
        n_ones_per_pat = n_ones_per_pat-edge_vector
        
    module_size_term = np.log(m_size+beta_K)
    # ones-matching
    oneRatios =(n_ones_per_pat+alpha/2)/(m_size+alpha)
    ones_matching_term = np.inner(np.log(oneRatios),edge_vector)
    # zero-matching
    zeroRatios = (m_size-n_ones_per_pat+alpha/2)/(m_size+alpha)
    zeros_matching_term = np.inner(np.log(zeroRatios),(1-edge_vector))
    lp = module_size_term+ones_matching_term+zeros_matching_term
    #q = moduleOneFreqs[edge]
    return lp #- calc_norm_coef(p,q)

def set_initial_conditions(network,exprs, verbose = True):
    t_0 = time.time()
    
    # 1. the number of edges inside each component, initially 1 for each component
    moduleSizes=np.ones(len(network.edges()),dtype=np.int)
    
    # 2. a binary (int) matrix of size n by m that indicates the patients on the edges
    edge2Patients = []
    all_pats = list(exprs.columns.values)
    for edge in network.edges():
        n1,n2 = edge
        pats_in_module = network[n1][n2]["patients"]
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
        del data['patients']
        data['m'] = i
        data['e'] = i
        i+=1
    
    #4.
    edge2Module = range(0,len(network.edges()))
    
    #5. edgeOneFreqs 
    edgeOneFreqs = []
    n = edge2Patients.shape[0]
    for e in range(0,n):
        edgeOneFreqs.append(float(sum(edge2Patients[e,]))/edge2Patients.shape[1])
    
    #6. edgeOneFreqs
    moduleOneFreqs = copy.copy(edgeOneFreqs)
    
    print("Initial state created in",round(time.time()- t_0,1) , "s runtime", file = sys.stdout)
    return moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, edgeOneFreqs, moduleOneFreqs

def adjust_lp(log_probs,n_exp_orders=7):
    # adjusting the log values before normalization to avoid under-flow
    max_p = max(log_probs)
    # shift all probs to set max_prob less than log(max_np.float)  
    log_probs = log_probs - max_p
    # set to minimal values all which less then 'n_orders' lesser than p_max to zeroes
    probs = []
    for lp in log_probs:
        if lp >= - n_exp_orders:
            probs.append(np.exp(lp))
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
def sampling(network,edge2Module, edge2Patients,nOnesPerPatientInModules,moduleSizes,
             edgeOneFreqs, moduleOneFreqs, p0, alpha = 0.1, beta_K = 1.0,
             max_n_steps=100,n_steps_averaged = 20,n_points_fit = 10,tol = 0.1,
             n_steps_for_convergence = 5,verbose=True):
    t_ =  time.time()
    edge2Module_history = [copy.copy(edge2Module)]
    #n_edge_skip_history = [0]
    #max_edges_oscillating = int(max_frac_edges_oscillating*len(network.edges()))
    is_converged = False
    for step in range(1,max_n_steps):
        print("step",step)
        not_changed_edges = 0
        t_0 = time.time()
        t_1=t_0
        i = 1
        for n1,n2,data in network.edges(data=True):
            # adjust LogP and sample a new module
            p = adjust_lp(data['log_p'],n_exp_orders=7)
            curr_module = data['m']
            edge_ndx = data['e']
            new_module = np.random.choice(data['modules'], p = p) 

            # update network and matrices if necessary
            if new_module != curr_module:
                apply_changes(network,n1,n2,edge_ndx,curr_module,new_module,
                              edge2Patients, edge2Module, nOnesPerPatientInModules,moduleSizes,
                              edgeOneFreqs, moduleOneFreqs, p0,
                              alpha=alpha,beta_K=beta_K)
                
            else:
                not_changed_edges +=1#
            i+=1
            if i%10000 == 1:
                print(i,"edges processed","\t",1.0*not_changed_edges/len(network.edges()),"%edges not changed;",
                      round(time.time()- t_1,1) , "s runtime",file=sys.stdout)
                not_changed_edges=0
                t_1 = time.time()
        print("\tstep",step,1.0*not_changed_edges/len(network.edges()),"- % edges not changed;",
              "runtime",round(time.time()- t_0,1) , "s", file = sys.stdout)
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
                print("Total runtime",round(time.time()- t_ ,1) , "s", file = sys.stdout)
            return edge2Module_history, n_final_steps,n_skipping_edges,P_diffs
    
    n_final_steps = n_steps_for_convergence
    if verbose:
        print("The model did not converge after", step,"steps.", file = sys.stdout)
        print("Consensus of last",n_final_steps,"states will be taken")
        print("Total runtime",round(time.time()- t_ ,1) , "s", file = sys.stdout)
        
    return edge2Module_history,n_final_steps,n_skipping_edges,P_diffs

def apply_changes(network,n1,n2, edge_ndx,curr_module, new_module,
                  edge2Patients,edge2Module,nOnesPerPatientInModules,moduleSizes,
                  edgeOneFreqs, moduleOneFreqs,p0,
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
                                 edgeOneFreqs,moduleOneFreqs,p0,alpha=alpha,beta_K=beta_K)
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
                                         edgeOneFreqs,moduleOneFreqs,p0,
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
                          nOnesPerPatientInModules,moduleSizes, edgeOneFreqs, moduleOneFreqs, p0, 
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
                                  edgeOneFreqs, moduleOneFreqs,p0,
                                  alpha=alpha,beta_K=beta_K,no_LP_update = True)
            
    print(changed_edges, "edges changed their module membership after taking consensus.")
    return consensus_edge2module, network, edge2Patients, nOnesPerPatientInModules, moduleSizes, edgeOneFreqs, moduleOneFreqs

################################## 3. Post-processing ##########################################

def identify_opt_sample_set(n_ones, exprs,genes, min_n_samples=50):
    best_sample_set = []
    best_SNR = 0
    best_thr = -1
    freq_ones = 1.0*n_ones/len(genes)
    thresholds = sorted(list(set(freq_ones)),reverse=True)
    for thr in thresholds:
        ndx = np.where(freq_ones >= thr)[0]
        samples = exprs.iloc[:,ndx].columns.values
        if len(samples) > exprs.shape[1]/2: # stop when 1/2 samples included
            return best_sample_set, best_thr, best_SNR
        if len(samples) > min_n_samples:
            avgSNR =  bicluster_avg_SNR(exprs,genes=genes,samples=samples)
            if avgSNR > best_SNR: 
                best_SNR = avgSNR 
                best_sample_set = samples
                best_thr = thr
    

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

def merge_modules(bics,nOnesPerPatientInModules,moduleSizes,exprs,
                 min_sample_overlap = 0.5,min_acceptable_SNR_percent=0.9,min_n_samples=50, verbose = True):
    
    t_0 = time.time()
    
    SNRs = [bic["avgSNR"] for bic in bics]
    if verbose:
        print("Input:", len(bics),"modules to merge", file = sys.stdout)
    
    ############## prepare sample overlap matrix  #########
    pat_overlap = np.zeros((len(bics),len(bics)))
    for i in range(0,len(bics)):
        bic_i = bics[i]
        for j in range(0,len(bics)):
            if i !=j:
                bic_j = bics[j]
                shared_genes = len(bic_i["genes"].intersection(bic_j["genes"]))
                shared_pats = float(len(bic_i["samples"].intersection(bic_j["samples"])))
                mutual_overlap = min(shared_pats/len(bic_i["samples"]),shared_pats/len(bic_j["samples"]))
                pat_overlap[i,j] = mutual_overlap 
    
    ############ run merging #############
    closed_modules = []
    while len(bics) > 0:
        ndx = np.argmax(SNRs)
        bic = bics[ndx]
        if verbose:
            print("Grow module:",bic["id"],"avg.|SNR|",bic["avgSNR"], "samples:",len(bic["samples"]), "genes:",  len(bic["genes"]))
        best_candidate_ndx = -1
        best_new_SNR = min_acceptable_SNR_percent*SNRs[ndx]
        best_pat_set = bic["samples"]
        candidate_ndxs = np.where(pat_overlap[ndx,:] >= min_sample_overlap )[0]
        for candidate_ndx in candidate_ndxs:
            bic2 = bics[candidate_ndx]
            if verbose:
                print("\t trying module:",bic2["id"],"avg.|SNR|",bic2["avgSNR"], "samples:",len(bic2["samples"]), "genes:", len(bic2["genes"]), "sample overlap:",pat_overlap[ndx,candidate_ndx])
            new_pats, new_SNR = calc_new_SNR(bic, bic2 ,exprs, nOnesPerPatientInModules,moduleSizes,min_n_samples=min_n_samples)
            if verbose:
                print("\t\t new avg.|SNR|:",new_SNR, "new samples",len(new_pats),"passed:",new_SNR > best_new_SNR)
            if new_SNR > best_new_SNR:
                #print("\t\t set SNR:", best_new_SNR, "-->", new_SNR, "and select",candidate_ndx)
                best_new_SNR = new_SNR
                best_candidate_ndx  =  candidate_ndx
                best_pat_set = new_pats
        if best_candidate_ndx >= 0:
            if verbose:
                print("\tmerge",bic["id"],"+",bics[best_candidate_ndx]["id"], "new avg.|SNR|",best_new_SNR,"samples:",len(best_pat_set),"genes:",len(set(bic["genes"].union(set(bics[best_candidate_ndx]["genes"])))))
            # add bic2 to bic and remove bic2
            bics, SNRs, pat_overlap, nOnesPerPatientInModules, moduleSizes = add_bic(ndx, best_candidate_ndx,bics, set(best_pat_set), best_new_SNR, SNRs,pat_overlap,nOnesPerPatientInModules,moduleSizes)
            # continue - pick new ndx_max
        else:
            if verbose:
                print("no more candidates to merge for ", bic["id"])
            # close module if no candidates for merging found
            closed_modules.append(bic)
            if verbose:
                print(bic["id"],"----------- closed. ")
            # remove module from all data structures
            bics, SNRs,pat_overlap = remove_bic(ndx, bics, SNRs, pat_overlap)
    print("merging finished in",round(time.time()- t_0,2) , "s", file = sys.stdout)
    return closed_modules


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
    bic["samples"] =  new_pats
    bic["avgSNR"] = new_SNR
    # update bic in bics, SNRs,pat_overlap
    bics[ndx] = bic
    SNRs[ndx] = new_SNR
    for j in range(0,len(bics)):
        if j !=ndx:
            shared_genes = len(bic["genes"].intersection(bics[j]["genes"]))
            if shared_genes > 0:
                shared_pats = float(len(bic["samples"].intersection(bics[j]["samples"])))
                mutual_overlap = min(shared_pats/len(bic["samples"]),shared_pats/len(bics[j]["samples"]))
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

def calc_new_SNR(bic, bic2,exprs, nOnesPerPatientInModules,moduleSizes,min_n_samples=50):
    n_ones = nOnesPerPatientInModules[bic["id"],]+nOnesPerPatientInModules[bic2["id"],]
    m_size = moduleSizes[bic["id"],]+moduleSizes[bic2["id"],]
    genes  = set(bic["genes"]).union(bic2["genes"])
    pats, thr, avgSNR = identify_opt_sample_set(n_ones, exprs, genes, min_n_samples=min_n_samples)
    return pats, avgSNR

#### evaluation ### 
def bicluster_corrs(exprs,genes=[],samples=[]):
    if len(samples) > 0 :
        mat = exprs.loc[genes,samples]
    else:
        mat = exprs.loc[genes, :]
    corrs  = []
    for g1,g2 in itertools.combinations(genes,2):
        corr = np.corrcoef(mat.loc[g1,:].values, mat.loc[g2].values)[0][1]
        corrs.append(corr)
    return round(np.average(corrs),2), round(np.min(corrs),2), round(np.max(corrs),2)


