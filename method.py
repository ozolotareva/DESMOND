from __future__ import print_function
import sys,os

import numpy as np
from scipy.stats import hypergeom # ,fisher_exact
from fisher import pvalue
import pandas as pd
 
import networkx as nx # should be < 2
import ndex2.client
import ndex2

import itertools
import warnings
import time
import copy
import random

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d 
import seaborn as sns

###### Reading and preprocessing of input files ####################
def prepare_input_data(exprs_file, network_file, seeds = [], verbose = True, min_n_nodes = 3):
    '''1) Converts the network into undirected, renames nodes. 
    \n2) Keeps only genes presenting in the network and expression matrix and retains only large connected components with more than 10 nodes.
    \n3) Remove seeds absent in the expression matrix or network.'''
    ### read expressoin matrix
    exprs = pd.read_csv(exprs_file, sep = "\t",index_col=0)
    exprs_genes = exprs.index.values
    ### read seeds 
    if len(set(exprs_genes)) != len(exprs_genes):
        if verbose:
            print("Duplicated gene names", file=sys.stderr)
    if type(seeds) == list or type(seeds) == set:
        seeds = set(seeds)
    elif type(seeds) == str:
        # read from file 
        seeds = set(list(pd.read_csv(seeds,header=None)[0].values))
        
    #### read and prepare the network
    # assume ndex format
    network = ndex2.create_nice_cx_from_file(network_file)
    network = network.to_networkx()
    network = nx.Graph(network.to_undirected())
    # rename nodes 
    network  = rename_nodes(network)
    network_genes = network.nodes()
    ccs = list(nx.connected_component_subgraphs(network))
    if verbose:
        print("Input:\n","\texpressions:",len(exprs_genes),"genes x",len(set(exprs.columns.values)),"patients;",
              "\n\tnetwork:",len(network_genes),"genes,",len(network.edges()) ,"edges in",len(ccs),"connected components:", 
              "\n\tseeds:", len(seeds),file=sys.stderr)

    network_genes = network.nodes()
    #### compare genes in network and expression and prune if necessary  ###
    genes = set(network_genes).intersection(set(exprs_genes))
    if len(genes) != len(network_genes):
        # exclude unnecessary genes from the network
        network_edges = len(network.edges())
        network = nx.subgraph(network,genes)
        if verbose:
            print(len(network_genes)-len(genes), "network nodes without expressoin profiles and",
                  network_edges-len(network.edges()),"edges excluded",file = sys.stderr)
    if len(genes) != len(exprs_genes):
        # exclude unnecessary genes from the expression matrix
        exprs = exprs.loc[genes,:]
        if verbose:
            print(len(exprs_genes)-len(genes), "genes absent in the network excluded from the expression matrix", file = sys.stderr)
    # remove small CCs containing less than min_n_nodes
    genes = []
    ccs = list(nx.connected_component_subgraphs(network))
    for cc in ccs:
        if len(cc.nodes()) >= min_n_nodes:
            genes += cc.nodes()
    network = nx.subgraph(network,genes)
    exprs = exprs.loc[genes,:]
    ccs = list(nx.connected_component_subgraphs(network))
    
    ### if seeds provided, exclude seeds absent in the network or expression matrix ###
    if len(seeds) > 0:
        keep_seeds = seeds.intersection(genes)
        if len(seeds) != len(keep_seeds):
            if verbose:
                print(len(seeds) - len(keep_seeds ),"seeds not found in expression matrix or in the network excluded.",file = sys.stderr)
            seeds = keep_seeds        
        if verbose:
            print("Processed Input:\n","\texpressions:",len(exprs.index.values),"genes x",len(set(exprs.columns.values)),"patients;",
                  "\n\tnetwork:",len(network_genes),"genes ",len(network.edges()) ,"edges in",len(ccs),"connected components:", 
                  "\n\tseeds:", len(seeds),file=sys.stderr)
        return exprs, network, seeds
    else:
        if verbose:
            print("Processed Input:\n","\texpressions:",len(exprs.index.values),"genes x",len(set(exprs.columns.values)),"patients;",
                  "\n\tnetwork:",len(network_genes),"genes ",len(network.edges()) ,"edges in",len(ccs),"connected components:",file=sys.stderr)
        return exprs, network # no seeds

def rename_nodes(G,attribute_name="name"):
    rename_nodes = {}
    for n in G.node:
        rename_nodes[n] =  G.node[n][attribute_name]
        G.node[n]["NDEx_ID"] = n
    return nx.relabel_nodes(G,rename_nodes)

def print_network_stats(G,print_cc = True):
    ccs = sorted(nx.connected_component_subgraphs(G), key=len,reverse=True)
    if nx.is_directed(G):
        is_directed = "Directed"
    else:
        is_directed = "Undirected"
    print(is_directed,"graph with",len(ccs),"connected components; with",len(G.nodes()),"nodes and",len(G.edges()),"edges;")
    if print_cc and len(ccs)>1:
        i = 0
        for cc in ccs:
            i+=1
            print("Connected component",i,":",len(cc.nodes()),"nodes and",len(cc.edges()),"edges")
            
def shortest_paths_matrix(G,node_list,add_rand_nodes = 0):
    n_list = []
    for n in node_list:
        if  n in G.nodes():
            n_list.append(n)
        else:
            print(n, "not found in graph excluded.",file=sys.stderr)
    rand_nodes = list(np.random.choice(G.nodes(),size =add_rand_nodes))
    print("random_nodes_added:",rand_nodes)
    n_list += rand_nodes 
    sh_paths = {}
    for n in n_list:
        sh_paths[n] = {}
        for n2 in n_list:
            sh_paths[n][n2] = nx.shortest_path_length(G, source=n, target=n2)
    
    return(pd.DataFrame.from_dict(sh_paths))

def replace_edge_attrs(subnet,edge_attr="masked",query=True,target = False,verbose = True):
    # e.g. {"masked":False, "patients":set()}
    n_edges = 0
    for edge in subnet.edges():
        n1,n2 = edge
        if subnet[n1][n2][edge_attr] == query:
            subnet[n1][n2][edge_attr] = target
            n_edges +=1 
    if verbose:
        print(n_edges, "edge attributes changed", file=sys.stderr)
        
def count_emtpy_edges(subnet, thr = 0):
    n_edges = 0
    for edge in subnet.edges():
        n1,n2 = edge
        if len(subnet[n1][n2]["patients"]) <= thr:
            n_edges+=1
    return n_edges

def count_masked_edges(subnet):
    n_edges = 0
    for edge in subnet.edges():
        n1,n2 = edge
        if subnet[n1][n2]["masked"]:
            n_edges+=1
    return n_edges

def plot_patient_ditsribution_and_mask_edges(subnet,min_n_patients=10,title=""):
    n_pats = []
    for edge in subnet.edges():
        n1,n2 = edge
        pats = len(subnet[n1][n2]["patients"])
        # mask edges with not enough patients 
        if pats < min_n_patients:
            subnet[n1][n2]["masked"] = True
        n_pats.append(pats)
    
    tmp = plt.hist(n_pats,bins=50)
    tmp = plt.title(title)
    return subnet

#### Precompute a matrix of thresholds for RRHO #####
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
    print("Precomputing RRHO thresholds",time.time()-t_0,"s", file = sys.stderr)
    return rrho_thresholds

########################### Step 1 - assigning patietns to edges ##################################
def expression_profiles2nodes(subnet, exprs, direction):
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
        print("Nodes in subnetwork:",len(nodes),"of them with expression:",len(nodes_with_exprs),file=sys.stderr)
        subnet = subnet.subgraph(list(nodes_with_exprs)).copy()
    # keep only genes corresponding nodes in the expression matrix 
    e = exprs.loc[nodes_with_exprs,:].T
    print("Genes in expression matrix",e.shape, file=sys.stderr)
    
    # assign expression profiles to every node  
    node2exprs = {}
    # store only sorted sample names
    for gene in e.columns.values:
        node2exprs[gene] = e[gene].sort_values(ascending=ascending)
    
    ## set node attributes
    nx.set_node_attributes(subnet, 'exprs', node2exprs)
    print("expression_profiles2nodes()\truntime:",round(time.time()-t_0,5),"s",file=sys.stderr)
    return subnet

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

def partialRRHO_old(gene1,gene2,subnet,fixed_step=10,significance_thr=0.05, verbose = False):
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
                    p_val, enrichment, patients = hg_test(rlist1,rlist2,t_u,t_w,return_items=True)
                    if p_val <  significance_thr:
                        #if verbose:
                        #    print("step#",i,"t_u,t_v:",t_u,t_w,"p_value:",p_val,"enrichment:",enrichment,"size",len(patients) )
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

def top_n(p1,p2,n):
    '''p1, p2 - expression profiles sorted in descending order;
    \nn - top n patients to take. By default 1/2 of the first list. 
    \nReturns two sets of patients - with gene1 and gene2 co-upregulated and co-downregulates.'''
    top = set(p1[:n]).intersection(set(p2[:n]))
    #down = set(p1[-n:]).intersection(set(p2[-n:]))
    return top #up3, down

def assign_patients2edges(subnet, method="top_half",fixed_step=10,significance_thr=0.05, rrho_thrs = False):
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
            print("'method' must be one of 'top_half' or 'RRHO'.",file = sys.stderr)
        runtimes.append(time.time() - t_0)
        i+=1
        if i%1000 == 0:
            print("\t",i,"edges processed. Average runtime per edge:",np.mean(runtimes), file=sys.stderr )
        #if n1 in seeds2 and n2 in seeds2:
        #    print(edge, len(up), len(down))
        #edge2pats[edge] = up
        subnet[n1][n2]["patients"] = up
        subnet[n1][n2]["masked"] = False
    print("assign_patients2edges()\truntime, s:", round(time.time() - t0,4),"for subnetwork of",len(subnet.nodes()),"nodes",
          len(subnet.edges()),"edges and","","patients.", file = sys.stderr)
    #nx.set_edge_attributes(subnet, 'patients', edge2pats)
    return subnet

def load_subnetworks(infile_name):
    t0 = time.time()
    '''Reads subnetworks from file.'''
    # read from file
    network = nx.read_edgelist(infile_name)
    subnetworks = sorted(nx.connected_component_subgraphs(network), key=len,reverse=True)
    # split into CC
    print("load_subnetworks() runtime", round(time.time()-t0,2),"s", file =sys.stderr)
    return subnetworks

def save_subnetworks(subnetworks,outfile_name):
    t0 = time.time()
    '''Writes subnetwork with associated patients on edges.'''
    # make graph of n subnetworks
    to_save = []
    # modify patients: set -> list 
    for i in range(0,len(subnetworks)):
        subnet = subnetworks[i].copy()
        for n1,n2 in subnet.edges():
            subnet[n1][n2]["patients"] = list(subnet[n1][n2]["patients"])
        to_save.append(subnet)
    network = nx.algorithms.operators.all.compose_all(to_save)
    # save to file
    nx.write_edgelist(network,outfile_name, data=True)
    print("save_subnetworks() runtime", round(time.time()-t0,2),"s", file =sys.stderr)
    
    
######### Step 2 functions ##########
# setting initial conditions 
def set_initial_distribution(subnet, exprs, basename, out_dir):
    t_0 = time.time()
    
    # 1). list of non-empty edges
    # consider only edges with associated patients
    edges = []
    for edge in subnet.edges():
        n1,n2 = edge
        if not subnet[n1][n2]["masked"]:
            edges.append(edge)
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stderr)
    
    
    # 2). dict of neighbouring edge indices for every edge
    t_0 = time.time()
    edgeNeighbourhood_file = basename+".edgeNeighorhood.npy"
    
    if os.path.exists(out_dir+edgeNeighbourhood_file): # if the file exists, load
        edgeNeighorhood = np.load(out_dir+edgeNeighbourhood_file).item()
        print("loaded",edgeNeighbourhood_file,"file in",round(time.time()- t_0,2) , "s", file = sys.stderr)
    else: # if no file exists, create and save
        edgeNeighorhood = {}
        for i in range(0,len(edges)):
            edge = edges[i]
            n1, n2 = edge 
            edgeNeighorhood[i] = []
            for j in range(0,len(edges)):
                adj_edge = edges[j]
                if n2 in adj_edge or n2 in adj_edge:
                    edgeNeighorhood[i].append(j)
            if i%10000 == 0:
                print(i, "edges processed")
        np.save(out_dir+edgeNeighbourhood_file, edgeNeighorhood)
        print("created",edgeNeighbourhood_file,"file in",round(time.time()- t_0,2) , "s", file = sys.stderr)
        
    t_0 = time.time()
    # 3. vector of modules indexes of edges, initialized so that each edge is in a separate module
    edge2module = range(0,len(edges))
    # 4. the number of edges inside each component, initially 1 for each component
    moduleSizes=np.ones(len(edges))
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stderr)
    # 5. edges with associated patients
    edge2patients = []
    all_pats = list(exprs.columns.values)
    for edge in edges:
        n1,n2 = edge
        pats_in_module = subnet[n1][n2]["patients"]
        x = np.zeros(len(all_pats), dtype='int')
        i = 0
        for p in all_pats:
            if p in pats_in_module:
                x[i] = 1.0
            i+=1
        edge2patients.append(x)


    # a binary (int) matrix of size n by m that indicates the patients on the edges
    edge2patients = np.asmatrix(edge2patients)

    # a matrix of size K by m that stores the total number of ones per patient in each module, initially equal to 'edge2patients'
    nOnesPerPatientInModules = copy.copy(edge2patients)
    print(edge2patients.shape,nOnesPerPatientInModules.shape)
    print(round(time.time()- t_0,2) , "s runtime", file = sys.stderr)
    return edges, edgeNeighorhood, edge2module, moduleSizes,  edge2patients, nOnesPerPatientInModules


### sampling
def sampling(edges,  edgeNeighorhood, edge2module, edge2patients, 
             moduleSizes,nOnesPerPatientInModules, max_n_steps=100,alpha = 0.1, beta_K = 1.0, min_pletau_steps = 30):
    edge2module_history = []
    edges_skip_history = []
    n_step = 0 
    ## convergence_conditions
    max_edges_oscillating = 0.005*len(edges)
    max_log_float = np.log(np.finfo(np.float64).max)
    while n_step < max_n_steps:
        t_0 = time.time()
        n_step += 1
        changes = []
        warnings = 0
        # sample the component for each edge 'i'
        for i in range(0,len(edges)):
            # excluding the information of edge 'i'
            moduleSizes[edge2module[i]] = moduleSizes[edge2module[i]]-1 # decrease i-th module size by one
            nOnesPerPatientInModules[edge2module[i],] = nOnesPerPatientInModules[edge2module[i],]- edge2patients[i,] # substract ones 
            # computing the possible neighbor components for edge 'i', including the current componenet of 'i' itself
            neighborEdges = edgeNeighorhood[i] # return all adjacent edge indicecs and index of a query edge 
            neighborModules = list(set([edge2module[e] for e in edgeNeighorhood[i]])) # return indices of all adjacent components 
            # compute probabilities of membership
            probs = np.zeros(len(neighborModules))
            for m in range(0,len(neighborModules)):
                j = neighborModules[m]
                oneRatios = (nOnesPerPatientInModules[j,]+alpha/2)/(moduleSizes[j]+alpha)
                zeroRatios = 1-oneRatios
                # term for module size + term for module values
                module_size_term = np.log(moduleSizes[j]+beta_K)
                ones_matching_term = np.inner(np.log(oneRatios),edge2patients[i,])[0][0] #35
                zeros_matching_term = np.inner(np.log(zeroRatios),(1-edge2patients[i,]))[0][0]
                probs[m] = module_size_term+ones_matching_term+zeros_matching_term
            # adjusting the log values before normalization to avoid over-/under-flow
            max_prob = max(probs)
            # handle under-float
            max_prob = max_prob - max_log_float#- np.log(len(neighborModules)))
            if max_prob > max_log_float:
                probs = probs-max_prob

            if sum(probs) > max_log_float:
                probs = probs-max_log_float
            probs = np.exp(probs)
            total_prob = sum(probs)
            # sampling the module 
            newModule = np.random.choice(neighborModules, p = probs/total_prob) #50
            if newModule != edge2module[i]:
                #print("change", i,"edge:",edge2module[i],"-->",newModule)
                changes.append(i)
            # placing i-th edge into a new module
            edge2module[i] = newModule
            # add ones to module
            moduleSizes[edge2module[i]] = moduleSizes[edge2module[i]]+1
            nOnesPerPatientInModules[edge2module[i],] = nOnesPerPatientInModules[edge2module[i],]+edge2patients[i,]
        edge2module_history.append(copy.copy(edge2module))
        edges_skip_history.append(changes)
        print("step",n_step,"n_changes",len(changes))#, "n_warnings", warnings)
        print(round(time.time()- t_0,4) , "s runtime", file = sys.stderr)
        # check convergence condition
        cc = check_convergence_condition(edge2module_history,edges_skip_history,
                                     min_pletau_steps = min_pletau_steps,
                                     max_edges_oscillating = max_edges_oscillating)
        if cc: # stop sampling if true 
            edge2module_history_slice = edge2module_history[-min_pletau_steps:]
            print("The model converged after", n_step,"steps.", file = sys.stderr)
            return edge2module_history, edges_skip_history, edge2module_history_slice
    print("The model did not converged.", file = sys.stderr)
    return edge2module_history, edges_skip_history,  edge2module_history[-min_pletau_steps:]


### get consensus module membership
def get_consensus_module_membrship(edge2module_history_slice, edges):
    edge2module = []
    labels = np.asarray(edge2module_history_slice)
    # identify modules which edges ocsilate 
    for i in range(0,len(edges)):
        unique, counts = np.unique(labels[:,i], return_counts=True)
        if len(unique) >1:
            #freq = np.array(counts,dtype=float)/10#labels.shape[0]
            counts = np.array(counts)
            new_ndx = unique[np.argmax(counts)]
            if float(max(counts))/labels.shape[0] < 0.5: 
                print("Warning: less than 50% of time in the most frequent module\n\tedge:",i,
                      "counts:",counts,"\n\tlabels:" , ",".join(map(str,labels[:,i])) ,file= sys.stderr)
                # TODO: put such edges to any empty neighbouring module
            edge2module.append(new_ndx)
        else:
            edge2module.append(unique[0])
    return edge2module

### read trajectories and restore modules ###### 
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

### check_convergence conditions  ###
def check_convergence_condition(edge2module_history,edges_skip_history,min_pletau_steps = 10,max_edges_oscillating =500):
    if len(edge2module_history) <= min_pletau_steps:
        return False
    else:
        skips = edges_skip_history[-min_pletau_steps:]
        #print(min(map(len,skips)) - max(map(len,skips)),map(len,skips))
    if max(map(len,skips)) > max_edges_oscillating:
        return False
    else:
        if max(map(len,skips)) - min(map(len,skips)) <= 0.5* max_edges_oscillating: #0.1
            return True
        else:
            return False

### manipulations with modules ##########
def get_genes(mid,edge2module=[],edges=[]):
    ndx = [i for i, j in enumerate(edge2module) if j == mid]
    genes = []
    for edge in [edges[i] for i in ndx]:
        genes.append(edge[0])
        genes.append(edge[1])
    genes = list(set(genes))
    return genes

def bicluster_avg_SNR(pats, genes, exprs,how = "median", absolute= False):
    mat = exprs.loc[genes, :]
    all_pats = set(mat.columns.values)
    bic= mat.loc[:, pats]
    out = mat.loc[:, all_pats.difference(pats)]
    avg_SNR = calc_avg_SNR(bic,out, how = "median", absolute= absolute)
    return round(avg_SNR,3)

def calc_avg_SNR(bic,out, how = "median", absolute= False):
    # from http://software.broadinstitute.org/cancer/software/genepattern/blog/2012/09/30/using-comparativemarkerselection-for-differential-expression-analysis
    if how == "mean":
        SNR = (bic.mean(axis=1) - out.mean(axis=1))/(bic.std(axis=1)+out.std(axis=1))
    elif how == "median":
        SNR = (bic.median(axis=1) - out.median(axis=1))/(bic.std(axis=1)+out.std(axis=1))
    else: 
        print("how must be mean or median",file = sys.stderr)
        return np.Nan
    if absolute: 
        return np.mean(abs(SNR))
    return np.mean(SNR)

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