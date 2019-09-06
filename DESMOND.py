from __future__ import print_function
import sys,os

import numpy as np
from scipy.stats import hypergeom
from fisher import pvalue
import pandas as pd
 
import networkx as nx

import itertools
import warnings
import time
import datetime
import copy
import random
import argparse
import pickle
import math

sys.path.append(os.path.abspath("/home/olya/SFU/Breast_cancer/DESMOND/"))

from desmond_io import prepare_input_data, load_network
from method import get_consensus_modules


parser = argparse.ArgumentParser(description="""Searches for gene sets differentially expressed in an unknown subgroup of samples and connected in the network.""" , formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-e','--exprs', dest='exprs_file', type=str, help='Expression matrix with sample name in columns and gene in rows, tab-separated.', default='', required=True,metavar='exprs_zscores.tsv')
parser.add_argument('-n','--network', dest='network_file', type=str, help='Network in tab or NDEx2 format.', default='', required=True, metavar='network.cx')
parser.add_argument('-d','--direction', dest='direction', type=str, help='Direction of dysregulation: UP or DOWN', default='UP', required=False)
#parser.add_argument('-m','--method', dest='method', type=str, help='How to assign patients on edges: RHHO or top_halves', default='RRHO', required=False)
parser.add_argument('-basename','--basename', dest='basename', type=str, help='Output basename without extention. If no outfile name provided output will be set "results_hh:mm_dddd-mm-yy".', default='', required=False)
parser.add_argument('-o','--out_dir', dest='out_dir', type=str, help='Output directory.', default='.', required=False)
### sampling parameters ###
parser.add_argument('--alpha', dest='alpha', type=float, help='Alpha.', default=0.1, required=False)
parser.add_argument('--beta_K', dest='beta_K', type=float, help='Beta/K.', default=1.0, required=False)
parser.add_argument('--p_val', dest='p_val', type=float, help='Significance threshold for RRHO method.', default=0.0005, required=False)
parser.add_argument('--max_n_steps', dest='max_n_steps', type=int, help='Maximal number of steps.', default=200, required=False)
parser.add_argument('--n_steps_averaged', dest='n_steps_averaged', type=int, help='Number of last steps analyzed when checking convergence condition. Values less than 10 are not recommended.', default=20, required=False)
parser.add_argument('--n_steps_for_convergence', dest='n_steps_for_convergence', type=int, help='Required number of steps when convergence conditions is satisfied.', default=5, required=False)
parser.add_argument('--min_n_samples', dest='min_n_samples', type=int, help='Minimal number of samples on edge.', default=-1, required=False)
### merging and filtering parameters
parser.add_argument('--min_SNR', dest='min_SNR', type=float, help='SNR threshold for biclusters to consider.', default=0.75, required=False)
parser.add_argument('-J','--min_sample_overlap', dest='J', type=float, help='minimal Jaccard index for two candidates for merging', default=0.5, required=False)
parser.add_argument('-r','--n_runs', dest='n_runs', type=int, help='Number of sampling runs.', default=1, required=False)

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
          "\nmin_SNR:",args.min_SNR,
          "alpha:",args.alpha, 
          "\nbeta/K:",args.beta_K,
          "\ndirection:",args.direction,
          "\nmin_sample_overlap(J):",args.J,
          "\nmax_n_steps:",args.max_n_steps,
          "\nn_steps_averaged:",args.n_steps_averaged,
          "\nn_steps_for_convergence:",args.n_steps_for_convergence,
          "\nn_runs:",args.n_runs,
          "\n",file = sys.stdout)
    
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

suffix  = ".alpha="+str(args.alpha)+",beta_K="+str(args.beta_K)+",direction="+args.direction+",p_val="+str(args.p_val)+",min_SNR="+str(args.min_SNR)
if args.verbose:
    print("Will save output files to:",args.out_dir,
        "\n\tOutput prefix:", basename,
        "\n\tOutput suffix:", suffix, file = sys.stdout)

##########################  Read and preprocess input files: network (.cx or .tab) and expressions ##############################
exprs, network = prepare_input_data(args.exprs_file, args.network_file, verbose = args.verbose)

# simplifying probability calculations
max_log_float = np.log(np.finfo(np.float64).max)
n_exp_orders = 7 # ~1000 times 
# define minimal number of patients in a module
if args.min_n_samples == -1:
    args.min_n_samples = int(max(10,0.05*len(exprs.columns.values))) # set to max(10, 5% of the cohort) 

if args. verbose:
    print("Mininal number of samples in a module:",args.min_n_samples ,file=sys.stdout)

N = exprs.shape[1]
p0 = N*np.log(0.5)+np.log(args.beta_K)
match_score = np.log((args.alpha*0.5+1)/(args.alpha))
mismatch_score = np.log((args.alpha*0.5+0)/args.alpha)
bK_1 = math.log(1+args.beta_K)


########################## RRHO ######################
    
# first check if the network already exists and try loading it
network_with_samples_file =  args.out_dir+basename + ".direction="+args.direction+",p_val="+str(args.p_val)+",minSNR="+ str(args.min_SNR)+",min_ns="+str(args.min_n_samples)+".network.txt"

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
    network = assign_patients2edges(network, min_SNR=args.min_SNR,min_n_samples=args.min_n_samples,
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
from method import get_genes, identify_opt_sample_set, bicluster_avg_SNR
from desmond_io import write_bic_table
from method import merge_biclusters

if args.verbose:
    print("Compute initial conditions...",file = sys.stdout)           

##### seting initial model state #######


all_discovered_bics = []
for run in range(0,args.n_runs):
    
    ### Set initial model state ###
    network = load_network(network_with_samples_file, verbose = args.verbose)
    
    [moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs, network] = set_initial_conditions(network,exprs,
                                        p0,match_score,mismatch_score,bK_1, N,
                                        log_func=np.log,
                                        alpha = args.alpha, beta_K = args.beta_K,
                                        verbose = args.verbose)
    
    ### Sampling
    print("Start sampling #%s..."%run,file = sys.stdout) 
    t0 = time.time()
    edge2Module_history,n_final_steps,n_skipping_edges,P_diffs = sampling(network,exprs, edge2Module, edge2Patients, nOnesPerPatientInModules,moduleSizes,moduleOneFreqs, p0, match_score,mismatch_score, bK_1,log_func=np.log, alpha = args.alpha, beta_K = args.beta_K, max_n_steps=args.max_n_steps, n_steps_averaged = args.n_steps_averaged, n_points_fit = 10, tol = 0.1, n_steps_for_convergence = args.n_steps_for_convergence, edge_ordering = "corr", verbose=False)
    print("time:\tSampling #%s fininshed in %s s and %s steps." %( run, round(time.time()-t0,2), len(edge2Module_history)), file = sys.stdout)
    
# save full edge2Module history    #save_object([edge2Module_history,n_final_steps,n_skipping_edges,P_diffs],edge2Module_history_file)

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
    t0 = time.time()
    filtered_bics = []
    few_genes = 0
    wrong_sample_number = 0
    low_SNR = 0
    #print([x for x in moduleSizes if x !=0])
    for mid in range(0,len(moduleSizes)):
        if moduleSizes[mid]>1: # exclude biclusters with too few genes
            genes = get_genes(mid,consensus_edge2module,network.edges())
            bic = identify_opt_sample_set(genes, exprs,
                                          direction=args.direction,
                                          min_n_samples=args.min_n_samples)
            avgSNR = bic["avgSNR"]
            if avgSNR ==-1:  # exclude biclusters with too few samples
                wrong_sample_number+=1
            elif avgSNR < args.min_SNR: # exclude biclusters with low avg. SNR 
                low_SNR += 1
            else:
                bic["id"] = mid
                filtered_bics.append(bic)
        elif moduleSizes[mid]>0:
            few_genes += 1 
    print("time:\tIdentified optimal patient for run#%s sets in %s" % (run, round(time.time()-t0,2)))

    print("\tBiclusters with 1 gene:",few_genes, file = sys.stdout)
    print("\tBiclusters with not enough or too many samples:",wrong_sample_number, file = sys.stdout)      
    print("\tBiclusters with low avg. |SNR|:", low_SNR, file = sys.stdout)

    print("\nPassed biclusters with >= 2 genes and >= %s samples: %s"%(args.min_n_samples,len(filtered_bics)), file = sys.stdout)

    # save raw biclusters
    #write_bic_table(filtered_bics, result_file_name+".biclusters.tsv")

    #### Merge remaining biclusters 
    merged_bics = merge_biclusters(filtered_bics, exprs,
                                   min_n_samples=args.min_n_samples,
                                   direction=args.direction,
                                   min_SNR=args.min_SNR, 
                                   J_threshold=args.J,
                                   verbose = False)

    print("Biclusters remaining after merging:", len(merged_bics))
    for bic in merged_bics:
        bic["n_genes"] = len(bic["genes"])
        bic["n_samples"] = len(bic["samples"])
        print("\t".join(map(str,[bic["id"],bic["n_genes"],bic["n_samples"],
                                 bic["avgSNR"],bic["direction"]])))
    all_discovered_bics+=merged_bics

i =0
for bic in all_discovered_bics:
    bic["id"] = i
    i+=1

if args.n_runs > 1:    
    print("Biclusters in all runs before final merging:",len(all_discovered_bics))
    merged_bics = merge_biclusters(all_discovered_bics, exprs,
                                       min_n_samples=args.min_n_samples,
                                       direction=args.direction,
                                       min_SNR=args.min_SNR, 
                                       J_threshold=args.J,
                                       verbose = args.verbose)    

    print("Biclusters remaining after merging of all runs:", len(merged_bics))
    for bic in merged_bics:
        bic["n_genes"] = len(bic["genes"])
        bic["n_samples"] = len(bic["samples"])
        print("\t".join(map(str,[bic["id"],bic["n_genes"],bic["n_samples"],
                                 bic["avgSNR"],bic["direction"]])))

result_file_name = args.out_dir+args.basename+suffix+",n_runs="+str(args.n_runs)

write_bic_table(merged_bics,
                result_file_name+".merged_J="+str(args.J)+".biclusters.tsv")

print(result_file_name,file=sys.stdout)
print("Total runtime:",round(time.time()-start_time,2),file = sys.stdout)