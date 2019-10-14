from __future__ import print_function
import sys,os

import numpy as np
from fisher import pvalue
import pandas as pd
 
import networkx as nx

import itertools
import warnings
import time
import copy
import random
import argparse
import pickle
import math

#sys.path.append(os.path.abspath("/home/olya/SFU/Breast_cancer/DESMOND/"))

from desmond_io import prepare_input_data, load_network, print_network_stats
from method import get_consensus_modules,identify_opt_sample_set


parser = argparse.ArgumentParser(description="""Searches for gene sets differentially expressed in an unknown subgroup of samples and connected in the network.""" , formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-e','--exprs', dest='exprs_file', type=str, help='Expression matrix with sample name in columns and gene in rows, tab-separated.', default='', required=True,metavar='exprs_zscores.tsv')
parser.add_argument('-n','--network', dest='network_file', type=str, help='Network in tab or NDEx2 format.', default='', required=True, metavar='network.cx')
parser.add_argument('-d','--direction', dest='direction', type=str, help='Direction of dysregulation: UP or DOWN', default='UP', required=False)
parser.add_argument('-basename','--basename', dest='basename', type=str, help='Output basename without extention. If no outfile name provided output will be set "results_hh:mm_dddd-mm-yy".', default='', required=False)
parser.add_argument('-o','--out_dir', dest='out_dir', type=str, help='Output directory.', default='.', required=False)
### sampling parameters ###
parser.add_argument('--alpha', dest='alpha', type=float, help='Alpha.', default=0.5, required=False)
parser.add_argument('--beta_K', dest='beta_K', type=float, help='Beta/K.', default=1.0, required=False)
parser.add_argument('--p_val', dest='p_val', type=float, help='Significance threshold for RRHO method.', default=0.01, required=False)
parser.add_argument('--max_n_steps', dest='max_n_steps', type=int, help='Maximal number of steps.', default=200, required=False)
parser.add_argument('--n_steps_averaged', dest='n_steps_averaged', type=int, help='Number of last steps analyzed when checking convergence condition. Values less than 10 are not recommended.', default=20, required=False)
parser.add_argument('--n_steps_for_convergence', dest='n_steps_for_convergence', type=int, help='Required number of steps when convergence conditions is satisfied.', default=5, required=False)
parser.add_argument('--min_n_samples', dest='min_n_samples', type=int, help='Minimal number of samples on edge. If not specified, set to max(10,0.1*cohort_size).', default=0, required=False)
### merging and filtering parameters
parser.add_argument('-q','--SNR_quantile', dest='q', type=float, help='Quantile determining minimal SNR threshold.', default=0.1, required=False)
### save or load precomputed  gene clusters
parser.add_argument('--save-gc', dest='save_gc', action='store_true', help='Save gene clusters resulted from sampling', required=False)
parser.add_argument('--load-gc', dest='load_gc',default='', type=str, help='Start from gene clusters resulted from sampling in previous runs.', required=False)

### plot flag
parser.add_argument('--plot_all', dest='plot_all', action='store_true', help='Switches on all plotting.', required=False)
### if verbose 
parser.add_argument('--verbose', dest='verbose', action='store_true', help='', required=False)

########################## Step 1. Read and check inputs ###############################################
start_time = time.time()

args = parser.parse_args()

if args.verbose:
    print("NetworkX version:",nx.__version__, "; must be < 2.", file = sys.stdout)


if args.verbose:
    print("Expression:",args.exprs_file, 
          "\nNetwork:",args.network_file,
          "\n",file = sys.stdout)
    print("\nRRHO significance threshold:",args.p_val,
          "\nSNR_quantile:",args.q,
          "\nalpha:",args.alpha, 
          "\nbeta/K:",args.beta_K,
          "\ndirection:",args.direction,
          "\nmax_n_steps:",args.max_n_steps,
          "\nn_steps_averaged:",args.n_steps_averaged,
          "\nn_steps_for_convergence:",args.n_steps_for_convergence,
          "\n",file = sys.stdout)

if args.verbose and args.min_n_samples!=0:
    print("\nmin_n_samples:",args.min_n_samples,file = sys.stdout)    
#### where to write the results ####
# create directory if it does not exists 
if not os.path.exists(args.out_dir) and not args.out_dir == "." :
    os.makedirs(args.out_dir)
args.out_dir = args.out_dir + "/"

#### define basename for output files if not provided
if args.basename:
    basename = args.basename
else: 
    [date_h,mins] = str(datetime.datetime.today()).split(":")[:2]
    [date, hs] = date_h.split()
    basename = "results_"+hs+":"+mins+"_"+date 

suffix  = ".alpha="+str(args.alpha)+",beta_K="+str(args.beta_K)+",direction="+args.direction+",p_val="+str(args.p_val)+",q="+str(args.q)
if args.verbose:
    print("Will save output files to:",args.out_dir,
        "\n\tOutput prefix:", basename,
        "\n\tOutput suffix:", suffix, file = sys.stdout)

################ Read and preprocess input files: network (.cx or .tab) and expressions (.tab) ##############
exprs, network = prepare_input_data(args.exprs_file, args.network_file, verbose = args.verbose)

if args.min_n_samples==0:
    args.min_n_samples = max(10,int(0.1*exprs.shape[1]))
    if args. verbose:
        print("Mininal number of samples in a module:",args.min_n_samples ,file=sys.stdout)

#### change gene and sample names to ints 
g_names2ints  = dict(zip(exprs.index.values, range(0,exprs.shape[0])))
ints2g_names = exprs.index.values
s_names2ints = dict(zip(exprs.columns.values, range(0,exprs.shape[1])))  
ints2s_names = exprs.columns.values
exprs.rename(index = g_names2ints, columns = s_names2ints,inplace=True)
#print(len(set(exprs.index.values)),len(set(exprs.columns.values)))
network=nx.relabel_nodes(network,g_names2ints)
#print_network_stats(network)


from sklearn.cluster import KMeans

avgSNR = []
t0 = time.time()
edges = network.edges()
lens = []
for i in range(0, 1000):
    g1,g2 = random.choice(edges)
    e = exprs.loc[[g1,g2],:]
    labels = KMeans(n_clusters=2, random_state=0,n_init=1,max_iter=100).fit(e.T).labels_
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1),len(ndx0))< args.min_n_samples:
        avgSNR.append(0)
    e1 = e.iloc[:,ndx1]
    e0 = e.iloc[:,ndx0]
    SNR0 = np.mean(e0.mean(axis=1)-e1.mean(axis=1)/(e0.std(axis=1)+e1.std(axis=1)))
    if np.isnan(SNR0):
        SNR0 = 0 
    avgSNR.append(abs(SNR0))

min_SNR = np.quantile(avgSNR,q=1-args.q)
print("avg.SNR threshold:",min_SNR)


### if load_gc is on try loading gene clusters
gene_clusters = []
if args.load_gc:
    print("Uses gene clusters from:",args.load_gc,file = sys.stdout)
    with open(args.load_gc) as infile:
        for line in infile.readlines():
            genes = line.rstrip().split(" ")
            #if type(g_names2ints.keys()[0])==int:
            #    genes = [g_names2ints[x] for x in genes]
            #else:
            genes = [g_names2ints[x] for x in genes]
            gene_clusters.append(genes)
else:
    # simplifying probability calculations
    max_log_float = np.log(np.finfo(np.float64).max)
    n_exp_orders = 7 # ~1000 times 
    # define minimal number of patients in a module
    if args.min_n_samples == -1:
        args.min_n_samples = int(max(10,0.05*len(exprs.columns.values))) # set to max(10, 5% of the cohort) 

    N = exprs.shape[1]
    p0 = N*np.log(0.5)+np.log(args.beta_K)
    match_score = np.log((args.alpha*0.5+1)/(args.alpha))
    mismatch_score = np.log((args.alpha*0.5+0)/args.alpha)
    bK_1 = math.log(1+args.beta_K)


    ########################## RRHO ######################

    # first check if the network already exists and try loading it
    network_with_samples_file =  args.out_dir+basename + ".direction="+args.direction+",p_val="+str(args.p_val)+",min_ns="+str(args.min_n_samples)+".network.txt"


    if os.path.exists(network_with_samples_file):
        from desmond_io import print_network_stats
        network = load_network(network_with_samples_file, verbose = args.verbose)
        print("Loaded annotated network from",network_with_samples_file,file = sys.stdout)
        print_network_stats(network)
        
    else:
        from method import precompute_RRHO_thresholds,  expression_profiles2nodes, assign_patients2edges, mask_empty_edges
        from desmond_io import save_network
        ##### assign expression vectors on nodes 
        network = expression_profiles2nodes(network, exprs, args.direction)

        # define step for RRHO
        fixed_step = int(max(1,0.01*len(exprs.columns.values))) # 5-10-20 ~15
        if args. verbose:
            print("Fixed step for RRHO selected:", fixed_step, file =sys.stdout)
        rrho_thresholds = precompute_RRHO_thresholds(exprs, fixed_step = fixed_step,significance_thr=args.p_val)

        ####  assign patients on edges
        network = assign_patients2edges(network, min_SNR=min_SNR,min_n_samples=args.min_n_samples,
                                           fixed_step=fixed_step, rrho_thrs = rrho_thresholds,verbose=args.verbose)

        # get rid of empty edges
        network = mask_empty_edges(network,min_n_samples=args.min_n_samples,remove=True, verbose=args.verbose)
        if args.plot_all:
            from desmond_io import plot_edge2sample_dist
            plot_outfile=args.out_dir + args.basename +suffix+".n_samples_on_edges.svg"
            plot_edge2sample_dist(network,plot_outfile)
        # get rid of components containing just 1 or 2 nodes 

        # save the network with patients on edges 
        save_network(network, network_with_samples_file, verbose = args.verbose)
        if args.verbose:
            print("Write network with samples to",network_with_samples_file,file= sys.stdout)

    ####################### Step 2. Sample module memberships ######################
    from method import set_initial_conditions
    from method import sampling
    from method import get_genes

    ### Set initial model state ###
    network = load_network(network_with_samples_file, verbose = args.verbose)
    
    if args.verbose:
        print("Compute initial conditions...",file = sys.stdout)           

    [moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs, network] = set_initial_conditions(network,exprs,
                                        p0,match_score,mismatch_score,bK_1, N,
                                        log_func=np.log,
                                        alpha = args.alpha, beta_K = args.beta_K,
                                        verbose = args.verbose)

    ### Sampling
    print("Start sampling ...",file = sys.stdout) 
    t0 = time.time()
    edge2Module_history,n_final_steps,n_skipping_edges,P_diffs = sampling(network,exprs, edge2Module, edge2Patients, nOnesPerPatientInModules,moduleSizes,moduleOneFreqs, p0, match_score,mismatch_score, bK_1,log_func=np.log, alpha = args.alpha, beta_K = args.beta_K, max_n_steps=args.max_n_steps, n_steps_averaged = args.n_steps_averaged, n_points_fit = 10, tol = 0.1, n_steps_for_convergence = args.n_steps_for_convergence, edge_ordering = "corr", verbose=args.verbose)
    print("time:\tSampling fininshed in %s s and %s steps." %( round(time.time()-t0,2), len(edge2Module_history)), file = sys.stdout)

    if args.plot_all:
        from desmond_io import plot_convergence
        plot_outfile = args.out_dir + args.basename +suffix+",ns_max=" + str(args.max_n_steps)+ ",ns_avg=" + str(args.n_steps_averaged) + ",ns_c="+str(args.n_steps_for_convergence) + ".convergence.svg"
        plot_convergence(n_skipping_edges, P_diffs,len(edge2Module_history)-n_final_steps,
                         args.n_steps_averaged, outfile=plot_outfile)

    ### take the last (n_points_fit+n_steps_for_convergence) steps
    edge2Module_history = edge2Module_history[-n_final_steps:]

    ### get consensus edge-to-module membership
    consensus_edge2module, network, edge2Patients, nOnesPerPatientInModules, moduleSizes, moduleOneFreqs = get_consensus_modules(edge2Module_history, network, edge2Patients, edge2Module, nOnesPerPatientInModules,moduleSizes, moduleOneFreqs, p0,match_score,mismatch_score, bK_1,log_func=np.log, alpha=args.alpha,beta_K=args.beta_K)

    print("Empty modules:", len([x for x in moduleSizes if x == 0]),
          "\nNon-empty modules:",len([x for x in moduleSizes if x > 0]))

################################## 3. Post-processing ########################
    few_genes = 0
    no_genes = 0
    for mid in range(0,len(moduleSizes)):
        if moduleSizes[mid]>1: # exclude biclusters with too few genes
            genes = get_genes(mid,consensus_edge2module,network.edges())
            gene_clusters.append(genes)
        elif moduleSizes[mid]>0:
            few_genes += 1 
        else:
            no_genes +=1
    print("Empty modules -- with 0 genes:",no_genes, file = sys.stdout)
    print("Biclusters with 1 gene:",few_genes, file = sys.stdout)
    print("Biclusters with >1 genes:",len(gene_clusters), file = sys.stdout)
    
    # save gene clusters to a file if necessary
    if args.save_gc:
        gc_file_name = args.out_dir + args.basename + ".alpha="+str(args.alpha)+ ",beta_K="+str(args.beta_K)+ ",direction="+args.direction+",p_val="+str(args.p_val)+ ".gene_clusters.txt"
        i =0 
        while True:
            if os.path.exists(gc_file_name):
                i+=1
                gc_file_name = args.out_dir+args.basename+suffix+".gene_clusters."+str(i)+".txt"
            else:
                print("Saves gene clusters with > 1 gene to:",gc_file_name ,file = sys.stdout)
                break
        fopen = open(gc_file_name,"w")
        for genes in gene_clusters:
            print(" ".join(map(str,[ints2g_names[gene] for gene in genes])), file = fopen)
        fopen.close()

        
### Make biclusters from gene clusters 
t0 = time.time()
filtered_bics = []
wrong_sample_number = 0
low_SNR = 0
bic_id = 0
for genes in gene_clusters:
    bic = identify_opt_sample_set(genes, exprs,direction=args.direction,min_n_samples=args.min_n_samples)
    avgSNR = bic["avgSNR"]
    if avgSNR ==-1:  # exclude biclusters with too few samples
        wrong_sample_number+=1
    elif avgSNR < min_SNR: # exclude biclusters with low avg. SNR 
        low_SNR += 1
    else:
        bic["id"] = bic_id
        bic_id+=1
        filtered_bics.append(bic)
    
print("time:\tIdentified optimal patient sets in %s" % (round(time.time()-t0,2)), file = sys.stdout)
print("\tBiclusters with not enough or too many samples:",wrong_sample_number, file = sys.stdout)      
print("\tBiclusters with low avg. |SNR|:", low_SNR, file = sys.stdout)

print("\nPassed biclusters with >= 2 genes, >= %s samples and avg.SNR>%s: %s"%(args.min_n_samples,round(min_SNR,2),len(filtered_bics)), file = sys.stdout)

#### Merge remaining biclusters 
from method import merge_biclusters
from desmond_io import write_bic_table

merged_bics = merge_biclusters(filtered_bics, exprs,
                               min_n_samples=args.min_n_samples,
                               direction=args.direction,
                               min_SNR=min_SNR, 
                               pval_threshold =0.05,
                               verbose = False)

print("Biclusters remaining after merging:", len(merged_bics))
for bic in merged_bics:
    bic["n_genes"] = len(bic["genes"])
    bic["n_samples"] = len(bic["samples"])
    bic["genes"] = { ints2g_names[x] for x in bic["genes"] }
    bic["samples"] = { ints2s_names[x] for x in bic["samples"] }
    print("\t".join(map(str,[bic["id"],bic["n_genes"],bic["n_samples"],
                             bic["avgSNR"],bic["direction"]])))


result_file_name = args.out_dir+args.basename+suffix

write_bic_table(merged_bics, result_file_name+".biclusters.tsv")

print(result_file_name,file=sys.stdout)
print("Total runtime:",round(time.time()-start_time,2),file = sys.stdout)
