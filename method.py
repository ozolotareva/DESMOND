from __future__ import print_function
import sys,os
import copy
import random
import warnings
import time

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, hypergeom
from fisher import pvalue
import itertools

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

import ndex2.client
import ndex2
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture



################## Network processsing and analysis ###############
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

def rename_nodes(G,attribute_name="name"):
    rename_nodes = {}
    for n in G.node:
        rename_nodes[n] =  G.node[n][attribute_name]
        G.node[n]["NDEx_ID"] = n
    return nx.relabel_nodes(G,rename_nodes)

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

###################### Step 1 - RWR ###########################
def set_p0(G, seeds):
    p0 = np.zeros(len(G.nodes()))
    for seed in seeds:
        i = G.nodes().index(seed)
        p0[i] = 1.0/(len(seeds))
    return p0

def run_RWR(gene,G,W_norm_r,r, delta = 0.000001):
    '''gene - starting point;\n
    G - networkx graph;\n
    W_norm_r - adjasency matrix, column-normalized and multiplied by (1-r) 
    r  - restart probability;\n
    delta - convergence threshold for L1 norm of the difference.'''
    p0 = set_p0(G,[gene])
    p_prev = p0
    diff_l1 = 1 
    n_steps = 1    
    while diff_l1 >= delta:
        p_next = np.asarray(W_norm_r.dot(p_prev))[0] + p0*r
        # L1 norm of the difference 
        diff_l1 = np.linalg.norm(np.subtract(p_next, p_prev), 1)
        if n_steps% 100 == 0:
            print("...","step =" ,n_steps,"diff L1",diff_l1, file = sys.stderr)
        n_steps +=1
        p_prev = p_next
    return p_next

def calc_distanse_matrix(seeds, G, W_norm_r, r, delta = 0.000001):
    '''calculates the matrix of "distances" (RWR probabilities) 
    from seed nodes to all other nodes of G'''
    t_0 = time.time()
    distance_matrix = {}
    for gene in seeds:
        if gene in G.nodes():
            distance_matrix[gene] = run_RWR(gene,G, W_norm_r,r,delta)
    distance_matrix = pd.DataFrame.from_dict(distance_matrix)
    distance_matrix.index = G.nodes()
    print(time.time() - t_0, "s distance matrix generation for",len(seeds),"seeds", file = sys.stderr)
    return distance_matrix


def group_seed_genes(G,distance_matrix, dist_thr, plot = True):
    '''Unites subnetworks induced by seed genes, if two seed genes are closer than dist_thr'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seed_genes = distance_matrix.columns.values
        r_G = nx.Graph()
        r_G.add_nodes_from(seed_genes)
        for gene in seed_genes:
            p_rw = distance_matrix[gene]
            p_rw = p_rw[p_rw > dist_thr]
            for partner in list(set(seed_genes).intersection(set(p_rw.index.values))):
                if gene != partner:
                    # dist is - 1/(np.log(RW prob.))  
                    dist = -1.0/np.log(p_rw[partner])
                    r_G.add_weighted_edges_from([(gene,partner,round(dist,2))])
        i= 0
        resulted_subnets = []
        for g in nx.connected_component_subgraphs(r_G):
            i+=1
            subnet_nodes = g.nodes()
            dm = distance_matrix.loc[:,subnet_nodes ]
            dm  = dm.loc[dm.max(axis =1)>dist_thr,:]
            print("component",i, "\nn_seeds",len(g.nodes()),"n_genes_in proximity",dm.shape[0])
            # get subnetwork and store in a list 
            resulted_subnets.append(G.subgraph(dm.index.values).copy())
            if plot and len(g.nodes())>2:
                pos = nx.spring_layout(g, iterations=1000)
                nx.draw(g,pos=pos, with_labels=True, font_weight='bold')
                rwdists = nx.get_edge_attributes(g,'weight')
                nx.draw_networkx_edge_labels(g,pos,edge_labels=rwdists)
                plt.show()
    return resulted_subnets

#################### Step 2 - functions ##############################
def expression_profiles2nodes(subnet, exprs, direction):
    t_0 = time.time()
    '''Associates sorted expression profiles with nodes. Removes nodes without expressions.
    \If direction="UP" sorts patinets in the descending order, othewise in ascending order - down-regulated first.'''
    if direction == "UP":
        ascending= False
    elif direction == "DOWN":
        ascending= true
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

def hh_test(rlist1,rlist2,t_u,t_w, return_items = False):
    N = len(rlist1)
    overlap = set(rlist1[:t_u]).intersection(set(rlist2[:t_w]))
    overlap_size = len(overlap)
    p_val = pvalue(overlap_size,t_u-overlap_size,t_w-overlap_size,N+overlap_size-t_u-t_w).right_tail
    enrichment = float(overlap_size)/(float((t_u)*(t_w))/N)
    if return_items:
        return p_val, enrichment, overlap
    else:
        return p_val, enrichment, overlap_size

def partialRRHO(gene1,gene2,subnet,fixed_step=10,significance_thr=0.05, verbose = False):
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
                    p_val, enrichment, patients = hh_test(rlist1,rlist2,t_u,t_w,return_items=True)
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

def assign_patients2edges(subnet, method="top_half"):
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
            up = partialRRHO(n1,n2,subnet,fixed_step=10,significance_thr=0.05,verbose=False)
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


def get_starting_edge(subnet, min_n_patients=50, to_mark=True):
    '''Return the unmasked edge with maximal number of associated patients. 
    Edges marked as "masked" exluded from consideration. If "to_mark" is True, marks edge as included.''' 
    max_pat = min_n_patients 
    strating_edge = False
    for edge in subnet.edges():
        n1,n2 = edge
        if not subnet[n1][n2]["masked"]:
            n_pats = len(subnet[n1][n2]["patients"])
            if n_pats > max_pat:
                max_pat  = n_pats
                strating_edge = edge 
    
    if strating_edge and to_mark:
        subnet[strating_edge[0]][strating_edge[1]]["masked"] = True
    return strating_edge, max_pat


def growCC_BFS(subnet,starting_edge,min_n_patients = 50, verbose = False):
    '''1) sets initial conditions:
    \n\t 1. adds starting_edge to CC
    \n\t 2. marks it as visites
    \n\t 3. intial patient set includes all patients associated with the edge 
    \n2) while number of associated patients is above the threshold, extends the CC
    '''
    t_0 = time.time()
    n1, n2 = starting_edge 
    CC = nx.from_edgelist([(n1, n2)])
    subnet[n1][n2]["masked"] = True
    # focus on the case of UP-regulated genes
    patients = subnet[n1][n2]["patients"] 
    best_edge = True
    # while number of associated patients is above the threshold, extend the CC
    while len(patients) > min_n_patients and best_edge:
        best_edge = False
        best_overlap = set() # best overlap on this iteration
        for n1,n2 in subnet.edges(CC.nodes()):
            if not subnet[n1][n2]["masked"]:
                pats = subnet[n1][n2]["patients"]
                shared = pats.intersection(patients)
                #print("\texamine edge",(n1,n2),"with patients",len(pats), len(shared),"shared.")
                if len(shared) > min_n_patients:
                    if len(shared) > len(best_overlap):
                        best_overlap = shared
                        best_edge = (n1,n2)
        if best_edge:
            # update patients 
            patients = best_overlap
            # update CC if any edge 
            CC.add_edge(*best_edge)
            # mark as visited 
            subnet[best_edge[0]][best_edge[1]]["masked"] = True
            if verbose:
                print("adding the edge",best_edge,"patients in ovelap",len(best_overlap))
            #print("visited edges:",visited_edges)
        else:
            if verbose:
                print("no edges found to extend CC with overlap above threhold of", min_n_patients,"patients. Close CC")
    if verbose:
        print("growCC_BFS() runtime:",round(time.time()-t_0,5),"s",file=sys.stderr)
    return CC, patients 


def find_all_CC(subnet, min_n_patients=50, outfile = "tmp.results.txt"):
    t_0 = time.time()
    CCs = {} # list of up-regulated Connected Components found
    # focus on the case of UP-regulated genes
    # pick unvisited edge with maximal number of patients above 'min_n_patients'
    starting_edge, pats = get_starting_edge(subnet,min_n_patients=50)
    #print("starting_edge",(n1,n2),"with",pats,"patients.")
    # while unviewed edges remain
    CC_id = 0 
    while starting_edge:
        # get connected component
        CC, patients  = growCC_BFS(subnet, starting_edge, min_n_patients = min_n_patients)
        # write the result to a file
        write_output(CC, patients, outfile)
        CCs[CC_id] = {"CC":CC,"patients": patients}
        CC_id +=1
        # report n components processed
        if CC_id% 500 == 0:
            print(CC_id,"CCs processed. Edges marked as used:",count_masked_edges(subnet),file=sys.stderr)
        #select a net starting edge if possible otherwise set False
        starting_edge,pats = get_starting_edge(subnet,min_n_patients=50)
    print("find_all_CC() runtime:",round(time.time()-t_0,5),"s",file=sys.stderr)
    return CCs

def write_output(CC, patients,outfile):
    pass


def growCC_BFS2(subnet,starting_edge,min_n_patients = 50, verbose = False):
    '''1) sets initial conditions:
    \n\t 1. adds starting_edge to CC and marks it as visited
    \n\t 3. intial patient set includes all patients associated with the edge 
    \n2) while number of associated patients is above the threshold, extends the CC
    '''
    t_0 = time.time()
    n1, n2 = starting_edge 
    CC = nx.from_edgelist([(n1, n2)])
    subnet[n1][n2]["masked"] = True
    # focus on the case of UP-regulated genes
    patients = subnet[n1][n2]["patients"] 
    best_edge = True
    # while number of associated patients is above the threshold, extend the CC
    while len(patients) > min_n_patients and best_edge:
        best_edge = False
        best_overlap = set() # best overlap on this iteration
        for n1,n2 in subnet.edges(CC.nodes()):
            if not subnet[n1][n2]["masked"]:
                pats = subnet[n1][n2]["patients"]
                shared = pats.intersection(patients)
                #print("\texamine edge",(n1,n2),"with patients",len(pats), len(shared),"shared.")
                if len(shared) > min_n_patients:
                    if len(shared) > len(best_overlap):
                        best_overlap = shared
                        best_edge = (n1,n2)
        if best_edge:
            # update patients 
            patients = best_overlap
            # update CC if any edge 
            CC.add_edge(*best_edge)
            # mark as visited 
            subnet[best_edge[0]][best_edge[1]]["masked"] = True
            # mark all edges connecting any pair of nodes in CC as visited 
            for n1,n2 in subnet.edges(CC.nodes()):
                if n1 in CC.nodes() and n2 in CC.nodes() and not CC.has_edge(n1,n2):
                    subnet[n1][n2]["masked"] = True
                    CC.add_edge(n1,n2)
            if verbose:
                print("adding the edge",best_edge,"patients in ovelap",len(best_overlap))
            #print("visited edges:",visited_edges)
        else:
            if verbose:
                print("no edges found to extend CC with overlap above threhold of", min_n_patients,"patients. Close CC")
    if verbose:
        print("growCC_BFS() runtime:",round(time.time()-t_0,5),"s",file=sys.stderr)
    return CC, patients 

def find_all_modules(subnet, min_n_patients=50, first_index = 0,min_n_genes = 4, verbose = True):
    t_0 = time.time()
    modules = {} # list of up-regulated Connected Components found
    # focus on the case of UP-regulated genes
    # pick unvisited edge with maximal number of patients above 'min_n_patients'
    starting_edge, pats = get_starting_edge(subnet,min_n_patients=50)
    #print("starting_edge",(n1,n2),"with",pats,"patients.")
    # while unviewed edges remain
    m_id = first_index
    while starting_edge:
        # get connected component
        CC, patients  = growCC_BFS2(subnet, starting_edge, min_n_patients = min_n_patients)
        # write the result to a file
        #write_output(CC, patients, outfile)
        if len(CC.nodes())> min_n_genes:
            modules[m_id] = {"CC":CC,"patients": patients}
        m_id +=1
        # report n components processed
        if m_id% 500 == 0 and verbose:
            print(m_id,"modules processed. Edges marked as used:",count_masked_edges(subnet),file=sys.stderr)
        #select a net starting edge if possible otherwise set False
        starting_edge,pats = get_starting_edge(subnet,min_n_patients=50)
    print("find_all_modules() runtime:",round(time.time()-t_0,5),"s",file=sys.stderr)
    return modules

################################ io ##############################3
def read_modules(infile_name):
    '''Reads modules from text file. Each module has id, patients, genes and edge. Returns list of modules.'''
    modules = {}
    ndx = False
    with open(infile_name, "r") as infile:
        for line in infile.readlines():
            line=line.rstrip()
            if line.startswith("#id:\t"):
                # close prev module
                if type(ndx)==int:
                    module["CC"] = nx.from_edgelist(edges)
                    modules[ndx] = module
                # open new module
                module = {}
                ndx = int(line.replace("#id:\t",""))
            if line.startswith("#genes:"):
                genes = line.replace("#genes:\t","").split(" ")
            if line.startswith("#patients:"):
                module["patients"] = line.replace("#patients:\t","").split(" ")
            if line.startswith("#edges:"):
                edges = line.replace("#edges:\t","").split(" ")
                edges = map(lambda x: tuple(sorted(x.split(","))),edges)
    modules[ndx] = module
    return modules

def write_modules(modules, outfile_name):
    '''Writes list of modules to file'''
    with open(outfile_name , "w") as outfile:
        for i in modules.keys():
            module = modules[i]
            outfile.write("#id:\t"+str(i)+"\n")
            pats = sorted(module["patients"])
            genes = sorted(module["CC"].nodes())
            edges = sorted(module["CC"].edges())
            outfile.write("#patients:\t"+ " ".join(pats)+"\n")
            outfile.write("#genes:\t"+ " ".join(genes)+"\n")
            outfile.write("#edges:\t"+ " ".join(map(lambda x:",".join(x),edges))+"\n")              
            

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
    #print(network["CCNE1"]["AR"].keys())
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
    
def keep_max_cc(ccs, network_genes, verbose = True):
    max_cc  = ccs[0]
    for cc in ccs[1:]:
        if len(cc.nodes()) > len(max_cc.nodes()):
            max_cc = cc
    if len(network_genes )!= len(max_cc.nodes()):
        if verbose:
            print("Removes small conneced components.",len(network_genes) -len(max_cc.nodes()),"nodes excluded.",file = sys.stderr)
        return max_cc
           
def prune_input_data(exprs_file, network_file, seeds = [], verbose = True):
    '''1) Convert network into undirected, rename nepodesm keep only large connected components > 10 nodes. 
    \n2) Keeps only genes presenting in the network and expression matrix.
    \n3) Remove seeds without expression or network.'''
    ### read expressoins
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
        
    #### read and prepare network
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
    # keep only largest CC
    max_cc = keep_max_cc(ccs,  network.nodes(), verbose = verbose)
    network = max_cc
    network_genes = network.nodes()
    #### compare genes in network and expression and prune if necessary 
    genes = set(network_genes).intersection(set(exprs_genes))
    if len(genes) != len(exprs_genes):
        exprs = exprs.loc[genes,:]
        if verbose:
            print(len(exprs_genes)-len(genes), "genes absent in the network excluded from expression matrix", file = sys.stderr)
    if len(genes) != len(network_genes):
        network_edges = len(network.edges())
        network = nx.subgraph(network,genes)
        if verbose:
            print(len(network_genes)-len(genes), "network nodes without expressoin profiles and",
                  network_edges-len(network.edges()),"edges excluded",file = sys.stderr)
    
    # keep only largest CC ramining after prooning
    ccs = list(nx.connected_component_subgraphs(network))
    max_cc = keep_max_cc(ccs,  network.nodes(), verbose = verbose)
    network = max_cc
    network_genes = network.nodes()
    genes = network_genes
    # exclude unnecessary genes from exprs
    exprs_genes = set(exprs.index.values)
    exprs = exprs.loc[genes,:]
    if verbose and len(exprs_genes)!=len(genes):
            print(len(exprs_genes)-len(genes), "genes absent in the network excluded from expression matrix", file = sys.stderr)
    # keep seeds 
    keep_seeds = seeds.intersection(genes)
    if len(seeds) != len(keep_seeds ):
        if verbose:
            print(len(seeds) - len(keep_seeds ),"seeds not found in expression matrix oe network excluded.",file = sys.stderr)
        seeds = keep_seeds        
    if verbose:
        ccs = list(nx.connected_component_subgraphs(network))
        print("Processed Input:\n","\texpressions:",len(exprs.index.values),"genes x",len(set(exprs.columns.values)),"patients;",
              "\n\tnetwork:",len(network_genes),"genes ",len(network.edges()) ,"edges in",len(ccs),"connected components:", 
              "\n\tseeds:", len(seeds),file=sys.stderr)
    return exprs, network, seeds


################################# Examining resluts ###################################
def compare_modules(m,m2):
    genes = set(m["CC"].nodes()).intersection(set(m2["CC"].nodes()))
    pats = len(m["patients"].intersection(m2["patients"]))
    if pats ==0 :
        min_percent_pat_overlap = 0
    else:
        p1 = len(m["patients"])
        p2 =  len(m2["patients"])
        min_percent_pat_overlap = min(float(pats)/p1, float(pats)/p2)
    return len(genes), pats, min_percent_pat_overlap 

def examine_modules(modules,exprs,min_genes=0,plot= True):
    stats = {}
    for m_id in modules.keys():
        m = modules[m_id]
        stats[m_id] = {}
        genes = m["CC"].nodes()
        pats = m["patients"]
        if len(genes) > min_genes:
            stats[m_id]["n_genes"] = len(genes)
            stats[m_id]["n_patients"] = len(pats)
            stats[m_id]["avgSNR"] = module_avg_SNR(m, exprs)
            avg_corr, min_corr, max_corr = module_corrs(m, exprs) 
            stats[m_id]["avg_corr"] = avg_corr
            stats[m_id]["max_corr"] = max_corr
            stats[m_id]["min_corr"] = min_corr
            #if avg_SNR > 0.6:
                #print(m_id,":","avg. SNR:",avg_SNR,"avg_r",avg_corr,"min_r",min_corr,len(patients),"patients",len(genes),"genes:"," ".join(sorted(genes)))
    stats = pd.DataFrame.from_dict(stats).T   
    if plot:        
        tmp = plt.figure(figsize=(20,5))
        tmp = plt.subplot(141)
        tmp = plt.hist(stats["n_genes"].values, bins =20)
        tmp = plt.title("Genes per module")
        tmp = plt.subplot(142)
        tmp = plt.hist(stats["n_patients"].values, bins =10)
        tmp = plt.title("Patients per module")
        tmp = plt.subplot(143)
        tmp = plt.hist(stats["avgSNR"].values, bins = 50)
        tmp = plt.title("Average SNR")
        tmp = plt.subplot(144)
        tmp = plt.scatter(stats["avgSNR"].values,stats["n_genes"].values)
        tmp = plt.xlabel("avg. SNR")
        tmp = plt.ylabel("n. genes in module")
        tmp = plt.title("genes vs avg. SNR; corr.:" +str(round(np.corrcoef(stats["avgSNR"].values,stats["n_genes"].values)[0][1],2)))
        # correlations of genes 
        tmp = plt.figure(figsize=(20,5))
        tmp = plt.subplot(131)
        tmp = plt.title("avg. r of genes in module")
        tmp = plt.hist(stats["avg_corr"].values, bins =20)
        tmp = plt.subplot(132)
        tmp = plt.title("min. r")
        tmp = plt.hist(stats["min_corr"].values, bins =20)
        tmp = plt.subplot(133)
        tmp = plt.title("max. r")
        tmp = plt.hist(stats["max_corr"].values, bins =20)
    return stats


def calc_avg_SNR(bic,out, how = "median",absolute = False):
    # from http://software.broadinstitute.org/cancer/software/genepattern/blog/2012/09/30/using-comparativemarkerselection-for-differential-expression-analysis
    if how == "mean":
        if absolute:
            SNR = (np.abs(bic.mean(axis=1) - out.mean(axis=1)))/(bic.std(axis=1)+out.std(axis=1))
        else:
            SNR = (bic.mean(axis=1) - out.mean(axis=1))/(bic.std(axis=1)+out.std(axis=1))
    elif how == "median":
        if absolute:
            SNR = (np.abs(bic.median(axis=1) - out.median(axis=1)))/(bic.std(axis=1)+out.std(axis=1))
        else:
            SNR = (bic.median(axis=1) - out.median(axis=1))/(bic.std(axis=1)+out.std(axis=1))
    else: 
        print("how must be mean or median",file = sys.stderr)
        return np.Nan
    return np.mean(SNR)

#### for modules 
def module_avg_SNR(module, exprs):
    pats = module["patients"]
    genes = module["CC"].nodes()
    mat = exprs.loc[genes, :]
    all_pats = set(mat.columns.values)
    bic= mat.loc[:, pats]
    out = mat.loc[:, ~mat.columns.isin(pats)]
    avg_SNR = calc_avg_SNR(bic,out, how = "median")
    return round(avg_SNR,3)

def module_corrs(module, exprs):
    genes = module["CC"].nodes()
    mat = exprs.loc[genes, :]
    corrs  = []
    for g1,g2 in itertools.combinations(genes,2):
        corr = np.corrcoef(mat.loc[g1,:].values, mat.loc[g2].values)[0][1]
        corrs.append(corr)
    return round(np.average(corrs),2), round(np.min(corrs),2), round(np.max(corrs),2)

### the same funcitons for biclusters ###
def bicluster_avg_FC(pats, genes, exprs):
    mat = exprs.loc[genes, :]
    all_pats = set(mat.columns.values)
    bic = mat.loc[:, pats]
    out = mat.loc[:, ~mat.columns.isin(pats)]
    avg_Fc = np.mean(bic.median(axis=1)/out.median(axis=1))
    return avg_Fc
def bicluster_avg_SNR(pats, genes, exprs,how = "median",absolute = False):
    mat = exprs.loc[genes, :]
    all_pats = set(mat.columns.values)
    #mat.shape, len(pats), fitness(mat,pats), len(all_pats)
    bic= mat.loc[:, pats]
    out = mat.loc[:, ~mat.columns.isin(pats)]
    avg_SNR = calc_avg_SNR(bic,out, how = "median",absolute = absolute)
    return round(avg_SNR,3)
def bicluster_corrs(genes, exprs, pats=[]):
    if len(pats) > 0 :
        mat = exprs.loc[genes, pats]
    else:
        mat = exprs.loc[genes, :]
    corrs  = []
    for g1,g2 in itertools.combinations(genes,2):
        corr = np.corrcoef(mat.loc[g1,:].values, mat.loc[g2].values)[0][1]
        corrs.append(corr)
    return round(np.average(corrs),2), round(np.min(corrs),2), round(np.max(corrs),2)

###################################  Optimization ########################
def plot_2DPCA(mat, pats, seed_pats = [], n_components=2):
    x = mat.T
    pca = PCA(n_components=n_components)
    pca.fit(x)
    pc = pca.transform(x)
    df = pd.DataFrame(pc, columns = ['principal component 1', 'principal component 2'], index=mat.columns.values)
    df.loc[:,"target"] = "out"
    df.loc[pats,"target"] = "opt_in"
    if len(seed_pats) > 0:
        df.loc[seed_pats,"target"] = "in"
    var_explained = pca.explained_variance_ratio_
    
    # plot 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1 '+str(round(var_explained[0],3)), fontsize = 15)
    ax.set_ylabel('PC 2 '+str(round(var_explained[1],3)), fontsize = 15)
    ax.set_title('PCA plot', fontsize = 20)

    targets = ['in','opt_in','out']
    colors = ['darkred','red', 'grey']
    markers = ['*','.','.']
    for target, color,marker in zip(targets,colors,markers):
        indicesToKeep = df['target'] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1']
                   , df.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 100 , marker = marker)
    ax.legend(targets)
    ax.grid()
    #return df
def get_bic_ndx(df, labels, pats):
    '''guess which index corresponds a bicluster now'''
    for ndx in set(labels):
        p = set(df.iloc[:,labels ==ndx].columns.values)
        # check if all patients in one cluster 
        if len(p.intersection(set(pats))) >= len(pats)*0.95:
            return ndx
    return -1

def kmeans_optimize(mat, pats , plot = True):
    labels = KMeans(n_clusters=2, n_init=10).fit_predict(mat.T)
    bic_ndx = get_bic_ndx(mat, labels, pats)
    if bic_ndx !=-1:
        opt_pats = mat.iloc[:,labels ==bic_ndx].columns.values
        if plot:
            plot_2DPCA(mat,opt_pats,seed_pats = pats, n_components=2)
        return opt_pats
    else:
        print("Failed to optimize",file=sys.stderr)
        return pats
    
def bgmm_optimize(mat, pats, plot = True):
    labels = BayesianGaussianMixture(n_components=2).fit(mat.T).predict(mat.T)
    bic_ndx = get_bic_ndx(mat, labels, pats)
    if bic_ndx !=-1:
        opt_pats = mat.iloc[:,labels ==bic_ndx].columns.values
        if plot:
            plot_2DPCA(mat,opt_pats,seed_pats = pats, n_components=2)
        return opt_pats
    else:
        print("Failed to optimize",file=sys.stderr)
        return pats

def gmm_optimize(mat, pats, plot = True):
    labels = GaussianMixture(n_components=2).fit(mat.T).predict(mat.T)
    bic_ndx = get_bic_ndx(mat, labels, pats)
    if bic_ndx !=-1:
        opt_pats = mat.iloc[:,labels ==bic_ndx].columns.values
        if plot:
            plot_2DPCA(mat,opt_pats,seed_pats = pats, n_components=2)
        return opt_pats
    else:
        print("Failed to optimize",file=sys.stderr)
        return pats
    
def plot_patient_ditsribution(subnet,thr,title="",min_n_patients=10):
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