from __future__ import print_function
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys,os
import random
import copy
import time

sys.path.append(os.path.abspath("/home/olya/SFU/Breast_cancer/DESMOND/"))
from method import print_network_stats

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
            fname_ext = ".ovelap=FF"
            bicluster_kind = "with no overlap"
        else:
            n_bics = min(len(bg_g)/genes_per_bic,max_n_bics)
            fname_ext = ".ovelap=FT"
            bicluster_kind = "with patient overlap"
    else:
        if p_overlap== False:
            n_bics = min(len(bg_p)/pats_per_bic,max_n_bics)
            fname_ext = ".ovelap=TF"
            bicluster_kind = "with gene overlap"
        else:
            n_bics =max_n_bics
            fname_ext = ".ovelap=TT"
            bicluster_kind = "gene and patient overlap"
            
    if n_bics != max_n_bics:
        print("Will create %s biclusters with %s instead of %s"%(n_bics, bicluster_kind, max_n_bics),file=sys.stderr)
    else:
        print("Will create %s biclusters with %s "%(n_bics,bicluster_kind),file=sys.stderr)
    
    biclusters = []

    # make patient annotation table
    anno = pd.DataFrame(index=map(lambda x : "bic"+str(x),range(0,n_bics)),
                        columns=df.columns.values,data=0).T
    anno.index.name = "patients2biclusters"
    # make patient annotation table
    anno_g = pd.DataFrame(index=map(lambda x : "bic"+str(x),range(0,n_bics)),
                        columns=df.index.values,data=0).T
    anno_g.index.name = "genes2biclusters"


    for bc in range(0,n_bics):
        # select random sets of patients and genes from the background
        genes  = list(np.random.choice(list(bg_g),size=genes_per_bic,replace=False))
        pats = list(np.random.choice(list(bg_p),size=pats_per_bic,replace=False))
        biclusters.append({"genes":genes, "patients":pats})
        bic_g+=genes
        bic_p+=pats
        # identify patients outside the bicluster
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
        df.index.name = "genes_patients"

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
            print("bic:",i,file=fopen)
            print("genes:","\t".join(map(str,bic["genes"])),file=fopen)
            print("patients:","\t".join(map(str,bic["patients"])),file=fopen)
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

def find_starting_points(G,bics,m=3,verbose = True):
    # identifies bics already assigned to G 
    # ignores biclusters not yet assigned to G
    # m is multiplicator for selecting nodes with degree m time s higher than the number of biclusters
    min_n_neigh = len(bics)
    consider_bics = set()
    for n,data in G.nodes(data=True):
        for bc in data['bics']:
            if bc in bics:
                consider_bics.add(bc)
    # identify seeds - nodes which neighbours already assigned to a bicluster
    if len(consider_bics) == 0:
        # if nothing to consider, add all empty nodes to starting points 
        starting_nodes = [node for node in G.nodes() if (len(G.node[node]['bics']) == 0 and len(G[node].keys()) >= min_n_neigh*m)]
        if verbose:
            print("Will consider no biclusters; de novo strating nodes identified",len(starting_nodes))
        return starting_nodes
    else:
        if verbose:
            print("Will consider biclusters",consider_bics,"previously added to the Graph")
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
            return []
        else:
            if verbose:
                print("Valid strating nodes identified",len(starting_nodes)) 
            return starting_nodes
        
def assign_nodes(G,genes, bics, verbose = True):
    min_n_neigh = len(bics)
    mapped_genes = {}
    
    starting_nodes =  find_starting_points(G,bics,m=3,verbose = verbose)

    G_ = G.copy() # not the same as !copy.copy(G)
    genes_ = copy.copy(genes)
    
    while len(starting_nodes)>0 and len(genes_)>0:
        # perform RW over G nodes until no unmapped genes remain or 
        # pick a random node with at least len(bics) neighbours
        g = genes_.pop()
        n = random.choice(starting_nodes)
        # add {node:gene} to the mapper
        mapped_genes[n] = g
        # update bics
        G_.node[n]['bics'] = bics
        # update adjacent nodes - all unassgned neigbours of mapped nodes
        adj_nodes = set()
        for n2 in mapped_genes.keys():
            adj_nodes.update([node for node in G_[n2].keys() if (len(G_.node[node]['bics']) == 0 and len(G_[node].keys()) >= min_n_neigh)])
        starting_nodes = list(adj_nodes)
    if len(genes_) == 0:
        return G_, mapped_genes
    else:
        return None
    
    
def grow_subnetworks(G,gene_groups,max_bics_per_gene, n_trials= 10, verbose=True):
    nx.set_node_attributes(G,'bics',[])
    mapping= {}
    all_genes = []
    for s in range(max_bics_per_gene,0,-1):
        for gene_group in gene_groups:
            bics = gene_group['bics']
            genes = copy.copy(gene_group['genes'])
            all_genes+=genes
            if len(bics) == s:
                if verbose:
                    print("Assing nodes for",len(genes),"gene(s) from", bics)
                n_trial = 1
                passed = False
                while n_trial < n_trials and not passed:
                    try:
                        G_, mapped_genes  = assign_nodes(G,genes, bics,verbose = verbose)
                        passed = True
                        if n_trial >1 and verbose:
                            print("%s attempt successfull"%n_trial)
                    except:
                        # try again ...
                        if verbose:
                            print("tries again for",len(genes),"gene(s) from bicluster(s)",bics)
                        n_trial +=1
                if passed:
                    # update mapping and G
                    mapping.update(mapped_genes)
                    G = G_
                else:
                    if verbose:
                        print("Failed for ",len(genes),"gene(s) from", bics,"Try again and increase n_trials.", file = sys.stderr)
                    return None
    # add background genes:
    unassigned_nodes = [node for node in G.nodes() if len(G.node[node]['bics']) == 0]
    if verbose:
        print("Background nodes",len(unassigned_nodes), file = sys.stderr)
    for gene_group in gene_groups:
        bics = gene_group['bics']
        genes = gene_group['genes']
        if len(bics) == 0:
            mapped_genes = dict(zip(unassigned_nodes,genes))
            break

    mapping.update(mapped_genes)
    if verbose:
        print("All nodes mapped:",set(mapping.keys())==set(G.nodes()), file = sys.stderr)
        print("All genes mapped:",set(mapping.values())==set(all_genes), file = sys.stderr)
    # relabel nodes
    G2 = nx.relabel_nodes(G,mapping=mapping)
    if verbose:
        print("All nodes named as sequence 0..(n_genes-1):",G2.nodes() == range(0,len(G2.nodes())), file = sys.stderr)
    return G2


def check_connectivity(G, anno,verbose = True):
    # checks whether genes from bicluster form a connected component
    all_connected = True
    for bc in range(0,anno.shape[1]):
        nodes_in_bic = []
        for n,data in G.nodes(data=True):
            if bc in data['bics']:
                nodes_in_bic.append(n)
        subnet = G.subgraph(nodes_in_bic)
        if len([nx.connected_components(G)])>1:
            all_connected = False
        if verbose:
            print_network_stats(subnet)
    return all_connected