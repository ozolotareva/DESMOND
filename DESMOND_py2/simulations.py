from __future__ import print_function
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys,os
import random
import copy
import time
import warnings
from fisher import pvalue


from desmond_io import print_network_stats

def make_bg_matrix(n_genes,n_pats):
    # generates empty matrix of (n_genes,n_pats)
    df = {}
    for g in range(0,n_genes):
        df[g] = np.random.normal(loc=0,scale=1.0,size = n_pats)
    df = pd.DataFrame.from_dict(df).T
    return df


def implant_biclusters(df,genes_per_bic=5,pats_per_bic=10,max_n_bics=1,bic_median=1.0,
                       outdir = "./",filename="implanted_bic", g_overlap=False,p_overlap=True):
    bic_g = []
    bic_p = []
    bg_g = set(df.index.values).difference(set(bic_g))
    bg_p = set(df.columns.values).difference(set(bic_p))
    if g_overlap == False:
        if p_overlap== False:
            n_bics = min(len(bg_g)/genes_per_bic,len(bg_p)/pats_per_bic,max_n_bics)
            fname_ext = ".overlap=FF"
            bicluster_kind = "with no overlap"
        else:
            n_bics = min(len(bg_g)/genes_per_bic,max_n_bics)
            fname_ext = ".overlap=FT"
            bicluster_kind = "with sample overlap"
    else:
        if p_overlap== False:
            n_bics = min(len(bg_p)/pats_per_bic,max_n_bics)
            fname_ext = ".overlap=TF"
            bicluster_kind = "with gene overlap"
        else:
            n_bics =max_n_bics
            fname_ext = ".overlap=TT"
            bicluster_kind = "gene and sample overlap"
            
    if n_bics != max_n_bics:
        print("Will create %s biclusters with %s instead of %s"%(n_bics, bicluster_kind, max_n_bics),file=sys.stderr)
    else:
        print("Will create %s biclusters with %s "%(n_bics,bicluster_kind),file=sys.stderr)
    
    biclusters = []

    # make samples annotation table
    anno = pd.DataFrame(index=map(lambda x : "bic"+str(x),range(0,n_bics)),
                        columns=df.columns.values,data=0).T
    anno.index.name = "samples2biclusters"
    # make samples annotation table
    anno_g = pd.DataFrame(index=map(lambda x : "bic"+str(x),range(0,n_bics)),
                        columns=df.index.values,data=0).T
    anno_g.index.name = "genes2biclusters"


    for bc in range(0,n_bics):
        # select random sets of samples and genes from the background
        genes  = list(np.random.choice(list(bg_g),size=genes_per_bic,replace=False))
        pats = list(np.random.choice(list(bg_p),size=pats_per_bic,replace=False))
        biclusters.append({"genes":genes, "samples":pats})
        bic_g+=genes
        bic_p+=pats
        # identify samples outside the bicluster
        if not g_overlap:
            bg_g = bg_g.difference(set(bic_g))
        if not p_overlap:
            bg_p = bg_p.difference(set(bic_p))
        # generate bicluster
        df_bic = {}
        for g in genes:
            df_bic[g] = dict(zip(pats,np.random.normal(loc=bic_median,
                                                       scale=1.0,size = pats_per_bic)))
        df_bic = pd.DataFrame.from_dict(df_bic).T   

        #implant the bicluster
        df.loc[genes,pats] = df_bic
        df.index.name = "genes_samples"

        # add record to annotation
        anno.loc[pats,"bic"+str(bc)] = 1
        anno_g.loc[genes,"bic"+str(bc)] = 1
    if filename:
        filename = filename+".N="+str(int(n_bics))+".Mu="+str(bic_median)+".GxP="+str(genes_per_bic)+","+str(pats_per_bic)+fname_ext
        df.to_csv(outdir+"/exprs/"+filename+".exprs.tsv",sep = "\t")
        # annotated df
        anno_g = anno_g.sort_values(by=list(anno_g.columns.values),ascending = False)
        anno = anno.sort_values(by=list(anno.columns.values),ascending = False)
        anno = anno.T
        df_anno = df.loc[anno_g.index.values, anno.columns.values]
        df_anno = pd.concat([anno_g,df_anno],axis =1)
        df_anno = pd.concat([anno,df_anno])
        df_anno = df_anno.loc[:,list(anno_g.columns.values)+list(anno.columns.values)]
        df_anno.to_csv(outdir+"/exprs_annotated/"+filename+".exprs_annotated.tsv",sep = "\t")
        fopen = open(outdir+"/true_biclusters/"+filename+".biclusters.txt","w")
        i=0
        for bic in biclusters:
            print("id:\t"+str(i),file=fopen)
            print("genes:\t"+" ".join(map(str,bic["genes"])),file=fopen)
            print("samples:\t"+" ".join(map(str,bic["samples"])),file=fopen)
            i+=1
        fopen.close()
    return df, biclusters,anno_g,filename

def simulate_network(n_genes, beta, delta,verbose = True):
    # makes scale-free networks
    t_0 = time.time()
    alpha = (1.0-beta)/2
    gamma = (1.0-beta)/2
    n_neigh =[]
    G = nx.scale_free_graph(n_genes,alpha=alpha,beta=beta,
                            gamma=gamma,delta_in=delta, delta_out=delta)
    if verbose:
        print("\tgamma:",1+(1.0+delta*(alpha+gamma))/(alpha+beta))
    # convert to undirected, with no self-loops or duplicated edges
    G = G.to_undirected()
    G.remove_edges_from(G.selfloop_edges())
    G = nx.Graph(G)
    # node degree distribution
    for n in G.nodes():
        n_neigh.append(len(G[n].keys()))
    tmp = plt.hist(n_neigh,range=(1,100),bins=50, density=True)
    plt.show()
    if verbose:
        print_network_stats(G)
        print("\tmin and max node degree: %s,%s"%(max(n_neigh), min(n_neigh)))
    print("runtime:",round(time.time()-t_0,2),"s",file = sys.stderr)
    return G


def group_genes(anno,verbose = True):
    # group genes together if they are asssigned to the same bicluster(s), e.g. all in 1 or all in 1,2,3 ... 
    colnames = list(anno.columns.values)
    anno["sum"] = anno.apply(sum,axis =1) # this is for sorting only
    anno.sort_values(["sum"]+colnames,ascending=False,inplace=True)
    anno = anno.loc[:,colnames]
    grouped = anno.groupby(colnames) # groupby by all columns
    max_bics_per_gene = 0
    gene_groups = []

    for name, group in grouped:
        #print(name,sum(name), len(group.index.values))
        max_bics_per_gene = max(max_bics_per_gene,sum(name))
        bics= []
        for i in range(0,len(name)):
            if name[i] == 1:
                bics.append(i)
        gene_groups.append({"genes":list(group.index.values),"bics":bics})
    if verbose:
        print("\tMax. number of biclusters per gene:",max_bics_per_gene)
        print("\tUnique gene groups",len(gene_groups))
    return gene_groups, max_bics_per_gene, anno 

def find_starting_points(G,bics,m=1,verbose = True):
    ''' identifies bics already assigned to G 
    \n\t m is multiplicator for selecting nodes with degree m times higher than the number of biclusters
    \n returns 1) a list of starting nodes  2) whether these nodes unlabelled'''
    
    min_n_neigh = len(bics)
    consider_bics = set()
    for n,data in G.nodes(data=True):
        for bc in data['bics']:
            if bc in bics:
                consider_bics.add(bc)
    # identify seeds - nodes which neighbours already assigned to a bicluster
    if len(consider_bics) == 0:
        # if nothing to consider, add all empty nodes to starting points 
        starting_nodes = [node for node in G.nodes() if (len(G.node[node]['bics']) == 0 and unassigned_order(node, G) > min_n_neigh*m)]
        if verbose:
            print("\twill consider no biclusters; de novo strating nodes identified",len(starting_nodes))
        return starting_nodes, True
    else:
        if verbose:
            print("\twill consider biclusters",consider_bics,"previously added to the Graph")
        consider_bics = list(consider_bics)
        starting_nodes = []
        for n in G.nodes():
            if len(G.node[n]['bics'])==0:
                bics_ = copy.copy(consider_bics)
                for neigh in G[n].keys():
                    assigned_bics = G.node[neigh]["bics"]
                    for bic in assigned_bics:
                        if bic in bics_:
                            bics_.remove(bic)
                    if len(bics_)==0:
                        # if all bics contacted by node n, add to starting points
                        starting_nodes.append(n)
                        break
        if len(starting_nodes) == 0:
            if verbose:
                print("Fails to identify any valid starting node. Try increasing m.", file = sys.stderr)
            return [], False
        else:
            if verbose:
                print("Valid strating nodes identified",len(starting_nodes)) 
            return starting_nodes, False

def unassigned_order(node, G):
    '''Counts the number of unlabelled neighbours for a node in G'''
    order = 0 
    for node2 in G[node].keys(): 
        if len(G.node[node2]['bics']) == 0:
            order +=1
    return order 

def DIAMOnD(n, G, CC, requred_len):
    '''Implements the method from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004120'''
    if len(CC)< requred_len:
        p_vals=[]
        s0 = len(CC)
        N = 2000
        best_pval,best_candidate = 1.0,-1
        for n_ in CC:
            for n2 in G[n_].keys():
                if len(G.node[n2]['bics']) == 0 and not n2 in CC:
                    n2_neigh = G[n2].keys()
                    ks =  len([x for x in n2_neigh if x in CC])# how many neighbours in CC
                    k_bg =  len(n2_neigh) - ks
                    p_val = pvalue(ks,k_bg,s0,N).right_tail
                    if p_val < best_pval:
                        best_pval = p_val
                        best_candidate = n2
        CC.append(best_candidate)
        if len(CC)< requred_len:
            CC=DIAMOnD(n, G, CC, requred_len)
    return CC

def DFS(n, G, CC, requred_len):
    '''Depth-first search'''
    for n2 in G[n].keys():
        if len(G.node[n2]['bics']) == 0 and not n2 in CC:
            if len(CC) < requred_len:
                CC.append(n2)
                CC = DFS(n2, G, CC, requred_len)
            else:
                return CC
    return CC

def BFS(n, G, CC, requred_len):
    '''Breadth-first search'''
    neighbours = []
    for n2 in G[n].keys():
        if len(G.node[n2]['bics']) == 0 and not n2 in CC:
            if len(CC) < requred_len:
                neighbours.append(n2)
                CC.append(n2)
            else:
                return CC
    for n2 in neighbours:
        if len(CC) < requred_len:
            CC = DFS(n2, G, CC, requred_len)
        else:
            return CC
    return CC

def assign_nodes(G,genes, bics,method="DFS", verbose = True):
    '''Assigns genes to G network nodes given gene-bicluster membership.
    Takes into account bicsluters which already mapped to the network. '''
    min_n_neigh = len(bics)
    mapped_genes = {}
    already_mapped = []
    if method == "DFS":
        func = DFS
    elif method == "BFS":
        func = BFS
    elif method == "DIAMOnD":
        func = DIAMOnD
    else:
        print("Wrong method name", method,file= sys.stderr)
        return None, {}
    
    # identifies seed nodes - which already assigned to a bicluster or pic a random unlabelled nodes
    starting_nodes, are_new = find_starting_points(G,bics,m=1,verbose = verbose)
    G_ = G.copy() 
    genes_ = copy.copy(genes)
    
    requred_len = len(genes_)
    
    random.shuffle(starting_nodes)
    #if not are_new: # 
    if verbose:
        print("\tGrow CC", bics, "from any of",len(starting_nodes),"nodes for", len(genes_),"genes.")
    for starting_node in starting_nodes:
        if are_new: # if new CC initialized, map starting node to a gene
            CC = [starting_node]
        else:
            CC = []
        requred_len = len(genes_)
        CC = func(starting_node, G_, CC, requred_len)
        # map genes
        for n in CC:
            g = genes_.pop()
            mapped_genes[n] = g
            G_.node[n]['bics'] = bics
        if len(genes_) == 0:
            return G_, mapped_genes
        else:
            if verbose:
                print("\t\tgenes remaining unassigned",len(genes_))
    print("\tFailed trial",trial,"for CC of",len(CC), "genes",len(genes_),file=sys.stderr )
    return None, {}
  

def grow_independent_subnetworks(G,gene_groups,max_bics_per_gene, method="DFS", verbose=True):
    mapping= {}
    for gene_group in gene_groups:
        bics = gene_group['bics']
        if len(bics) == 1:
            genes = copy.copy(gene_group['genes'])
            if verbose:
                print("Assing nodes for",len(genes),"gene(s) from", bics,"using", method)
            G, mapped_genes  = assign_nodes(G,genes, bics, method=method,verbose = verbose)
            if len(mapped_genes.keys())>0:
                mapping.update(mapped_genes)
            else:
                if verbose:
                    print("Failed mapping for ",len(genes),"gene(s) from", bics, file = sys.stderr)
                return None, {}
    return G, mapping
    

def add_shared_nodes(G,gene_groups,max_bics_per_gene, mapping, verbose=True):
    # add shared nodes, starting from nodes assigned to many bicslucters
    for s in range(max_bics_per_gene,1,-1):
        for gene_group in gene_groups:
            bics = gene_group['bics']
            genes = copy.copy(gene_group['genes'])
            if len(bics) == s:
                if verbose:
                    print("Assing nodes for",len(genes),"gene(s) from", bics)
                starting_nodes, are_new = find_starting_points(G,bics,m=1,verbose = verbose)
                n_genes = len(genes)
                if len(starting_nodes) >= n_genes:
                    nodes = random.sample(starting_nodes,n_genes)
                    for i in range(0,n_genes): # not necessarily directly connected
                        # update mapping and G
                        mapping[nodes[i]]=genes[i]
                        G.node[nodes[i]]['bics'] = bics
                    print("Success for",len(genes),"gene(s) from", bics)
                else:
                    print("Failed for ",len(genes),"gene(s) from", bics,".No free candidate nodes connecting subnetworks available.", file = sys.stderr)
                    return False, None
    
    assigned_nodes = [node for node in G.nodes() if len(G.node[node]['bics']) != 0]
    print("All genes from biclusters mapped:", set(assigned_nodes)== set( mapping.keys()))
    # add background genes:
    unassigned_nodes = [node for node in G.nodes() if len(G.node[node]['bics']) == 0]
    
    if verbose:
        print("Background nodes",len(unassigned_nodes))
    for gene_group in gene_groups:
        bics = gene_group['bics']
        genes = gene_group['genes']
        if len(bics) == 0:
            unassigned_genes = genes
            print("Unmapped genes:", len(unassigned_genes))
    mapping.update(dict(zip(unassigned_nodes,unassigned_genes)))
    if verbose:
        print("All nodes mapped:",set(mapping.keys())==set(G.nodes()), file = sys.stderr)
    # relabel nodes
    G = nx.relabel_nodes(G,mapping=mapping)
    
    return True, G

def mapp_all_nodes(G,gene_groups,max_bics_per_gene,  method="DFS", verbose=True):
    G_ = G.copy()
    nx.set_node_attributes(G_,'bics',[])
    nx.set_node_attributes(G_,'gene',[])
    # first create independent networks
    print("Grow independent networks ")
    G_, mapping = grow_independent_subnetworks(G_,gene_groups,max_bics_per_gene, method=method, verbose=verbose)
    if len(mapping)>0:
        print("\tAdd shared genes")
        # then select nodes connecting independent subnetworks and label with shared genes
        passed, G_ = add_shared_nodes(G_,gene_groups,max_bics_per_gene,mapping, verbose=verbose)
        if passed:
            return G_
        else:
            print("\t...failed assigning shared genes")
    else:
        print("\t...failed growing independent subnetworks")


def check_connectivity(G, anno,verbose = True, plot = False):
    # checks whether genes from bicluster form a connected component
    all_connected = True
    for bc in range(0,anno.shape[1]):
        nodes_in_bic = []
        for n,data in G.nodes(data=True):
            if bc in data['bics']:
                nodes_in_bic.append(n)
        subnet = G.subgraph(nodes_in_bic)
        if len([x for x in nx.connected_components(subnet)])>1:
            all_connected = False
            print("Subnetwork",bc, "is not connected.", file = sys.stderr)
        if plot:
            pos = nx.spring_layout(subnet , iterations=1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nx.draw(subnet ,pos=pos, with_labels=True, font_weight='bold')
                plt.show()
        if verbose:
            print_network_stats(subnet)
    return all_connected