from __future__ import print_function
import sys,os
import pandas as pd
import networkx as nx
import pickle


import time

import matplotlib
import matplotlib.pyplot as plt 

###### Reading and preprocessing of input files ####################
def prepare_input_data(exprs_file, network_file, verbose = True, min_n_nodes = 3):
    '''1) Reads network in .tab or cx format. Converts the network into undirected, renames nodes. 
    \n2) Keeps only genes presenting in the network and expression matrix and retains only large connected components with more than min_n_nodes nodes. '''
    ### read expressoin matrix
    exprs = pd.read_csv(exprs_file, sep = "\t",index_col=0)
    #exprs.rename(str,axis="index",inplace=True) # this is because 
    exprs_genes = exprs.index.values

    if len(set(exprs_genes)) != len(exprs_genes):
        if verbose:
            print("Duplicated gene names", file=sys.stderr)
        
    #### read and prepare the network
    # try reading .tab network, e.g.gene1\tgene2\t...
    try:
        network = nx.read_weighted_edgelist(network_file)
        try: 
            network = nx.relabel_nodes(network,int)
        except:
            print("Node names are string", file=sys.stderr)
    except:
        # assume ndex format
        import ndex2.client
        import ndex2
        network = ndex2.create_nice_cx_from_file(network_file)
        network = network.to_networkx()
        # remove unncessarry edge atributes 
        for n1,n2,data in network.edges(data=True):
            del data['interaction']
        # rename nodes 
        network  = rename_ndex_nodes(network)
    # convert to undirected network
    network = nx.Graph(network.to_undirected())
    network_genes = network.nodes()
    ccs = list(nx.connected_component_subgraphs(network))
    if verbose:
        print("Input:\n","\texpressions:",len(exprs_genes),"genes x",len(set(exprs.columns.values)),"patients;",
              "\n\tnetwork:",len(network_genes),"genes,",len(network.edges()) ,"edges in",len(ccs),"connected components:"
              ,file=sys.stdout)

    network_genes = network.nodes()
    
    #### compare genes in network and expression and prune if necessary  ###
    genes = set(network_genes).intersection(set(exprs_genes))
    if len(genes) != len(network_genes):
        # exclude unnecessary genes from the network
        network_edges = len(network.edges())
        network = nx.subgraph(network,genes)
        if verbose:
            print(len(network_genes)-len(genes), "network nodes without expressoin profiles and",
                  network_edges-len(network.edges()),"edges excluded",file = sys.stdout)
    if len(genes) != len(exprs_genes):
        # exclude unnecessary genes from the expression matrix
        exprs = exprs.loc[genes,:]
        if verbose:
            print(len(exprs_genes)-len(genes), "genes absent in the network excluded from the expression matrix", file = sys.stdout)
    # remove small CCs containing less than min_n_nodes
    genes = []
    ccs = list(nx.connected_component_subgraphs(network))
    for cc in ccs:
        if len(cc.nodes()) >= min_n_nodes:
            genes += cc.nodes()
    network = nx.subgraph(network,genes)
    exprs = exprs.loc[genes,:]
    ccs = list(nx.connected_component_subgraphs(network))
    
    
    if verbose:
        print("Processed Input:\n","\texpressions:",len(exprs.index.values),"genes x",len(set(exprs.columns.values)),"patients;",
                  "\n\tnetwork:",len(network_genes),"genes ",len(network.edges()) ,"edges in",len(ccs),"connected components:",file=sys.stdout)
        return exprs, network 

def rename_ndex_nodes(G,attribute_name="name"):
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
    print(is_directed,"graph with",len(ccs),"connected components; with",len(G.nodes()),"nodes and",len(G.edges()),"edges;", file = sys.stdout)
    if print_cc and len(ccs)>1:
        i = 0
        for cc in ccs:
            i+=1
            print("Connected component",i,":",len(cc.nodes()),"nodes and",len(cc.edges()),"edges", file = sys.stdout)

            
############### save and load temporary results #######################

def load_network(infile_name, verbose = True):
    t0 = time.time()
    '''Reads subnetworks from file.'''
    # read from file
    network = nx.read_edgelist(infile_name)
    try:
        network = nx.relabel_nodes(network,int)
    except:
        pass
    if verbose:
        print("load_network() runtime", round(time.time()-t0,2),"s", file =sys.stdout)
    return network

def save_network(network,outfile_name, verbose = True):
    t0 = time.time()
    '''Writes subnetwork with associated patients on edges.'''
    # make graph of n subnetworks
    to_save = []
    # modify patients: set -> list 
    for n1,n2 in network.edges():
        network[n1][n2]["patients"] = list(network[n1][n2]["patients"])
    # save to file
    nx.write_edgelist(network,outfile_name, data=True)
    if verbose:
        print("save_network() runtime", round(time.time()-t0,2),"s", file =sys.stdout)

######### save and load initial state data #########
def save_object(obj, filename):
    t_0 = time.time()
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print("saved data in file",filename,round(time.time()- t_0,1) , "s", file = sys.stdout)
    
def load_object(filename):
    t_0 = time.time()
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    print("loaded data in from file",filename,round(time.time()- t_0,1) , "s", file = sys.stdout)
    return obj

###### save modules ####################
def write_modules(bics,file_name):
    fopen = open(file_name,"w")
    for bic in bics:
        print("id:\t"+str(bic["id"]), file=fopen)
        print("average SNR:\t"+str(bic["avgSNR"]),file=fopen)
        print("genes:\t"+" ".join(map(str,bic["genes"])),file=fopen)
        print("samples:\t"+" ".join(map(str,bic["samples"])),file=fopen)
    fopen.close()
    print(str(len(bics)),"modules written to",file_name,file = sys.stdout)
    
    
### plots numnber of oscilating edges and RMS(Pn-Pn+1)
def plot_convergence(n_skipping_edges,P_diffs, thr_step,n_steps_averaged, outfile = ""):
    steps = range(n_steps_averaged,n_steps_averaged+len(n_skipping_edges))
    fig, axarr = plt.subplots(2, 1,sharex=True, figsize=(15,7))
    axarr[0].set_title("Model convergence")
    axarr[0].plot(steps, n_skipping_edges,'b.-')
    axarr[0].axvline(thr_step,color="red",linestyle='--') 
    axarr[0].set_ylabel("# edges oscilating on the last "+str(int(n_steps_averaged))+" steps")
    steps = range(n_steps_averaged,n_steps_averaged+len(P_diffs))
    axarr[1].plot(steps,P_diffs,'b.-' )
    axarr[1].set_xlabel('step')
    axarr[1].axvline(thr_step,color="red",linestyle='--') 
    tmp = axarr[1].set_ylabel("RMS(Pn-Pn+1)")
    if outfile:
        plt.savefig(outfile, transparent=True)

### plots the distribution of number of samples over all populated edges          
def plot_edge2sample_dist(network,outfile):
    n_samples = []
    for edge in network.edges():
        n1,n2 = edge
        samples = len(network[n1][n2]["patients"])
        n_samples.append(samples)
        # mask edges with not enough samples
    tmp = plt.hist(n_samples,bins=50)
    tmp = plt.title("Distribution of samples associated with edges.")
    plt.savefig(outfile, transparent=True)
    
def plot_bic_stats(bics, outfile):
    tmp = plt.figure(figsize=(20,5))
    i = 1
    for var in ["genes", "samples"]:
        vals = []
        for bic in bics:
            vals.append(len(bic[var]))
        plt.subplot(1,3,i)
        i+=1
        tmp = plt.hist(vals, bins=50)
        tmp = plt.title(var)
    vals = []
    plt.subplot(1,3,3)
    for bic in bics:
        vals.append(bic["avgSNR"])
    tmp = plt.hist(vals, bins=50)
    tmp = plt.title("avg. |SNR|")
    plt.savefig(outfile, transparent=True)
    
