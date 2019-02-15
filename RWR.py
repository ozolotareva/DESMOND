from __future__ import print_function
import sys,os
import time
import numpy as np 
import pandas as pd
import warnings
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt 


def create_subnetworks_from_seeds(network,seeds,r,delta,RWR_probability_thr, verbose = True):
    t_0 = time.time()
    # this is quite slow, but we need this just once
    W = nx.adjacency_matrix(network).todense()
    # let's compute W_norm*(1-r) - row-wise normalized matrix multiplied by prob. to continue (1-r)
    W_norm_r  = np.multiply(W,1.0-r)/W.sum(axis=0)
    print(time.time() - t_0, "s for adjacency matrix generation", file = sys.stderr)
    ### Calculate RWR-based distances between all connected genes
    distance_matrix = calc_distanse_matrix(seeds, network, W_norm_r, r, delta = delta)
    ### Merge subnetwork which share seed genes 
    resulted_subnets = group_seed_genes(network,distance_matrix, dist_thr=RWR_probability_thr, plot = True)
    if verbose:
        print("Will search for modules in ",len(resulted_subnets),"following subnetworks", file = sys.stderr)
        for subnet in resulted_subnets:
            n_nodes = len(subnet.nodes())
            print("nodes:",n_nodes,"edges:",len(subnet.edges()), file = sys.stderr)
    return resulted_subnets 


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