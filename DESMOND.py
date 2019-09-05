from __future__ import print_function
# v1.2
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

from desmond_io import prepare_input_data
from method import get_consensus_modules


parser = argparse.ArgumentParser(description="""Searches for gene sets differentially expressed in an unknown subgroup of samples and connected in the network.""" , formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-e','--exprs', dest='exprs_file', type=str, help='Expression matrix with sample name in columns and gene in rows, tab-separated.', default='', required=True,metavar='exprs_zscores.tsv')
parser.add_argument('-n','--network', dest='network_file', type=str, help='Network in tab or NDEx2 format.', default='', required=True, metavar='network.cx')
parser.add_argument('-d','--direction', dest='direction', type=str, help='Direction of dysregulation: UP or DOWN', default='UP', required=False)
parser.add_argument('-m','--method', dest='method', type=str, help='How to assign patients on edges: RHHO or top_halves', default='RRHO', required=False)
parser.add_argument('-basename','--basename', dest='basename', type=str, help='Output basename without extention. If no outfile name provided output will be set "results_hh:mm_dddd-mm-yy".', default='', required=False)

parser.add_argument('-o','--out_dir', dest='out_dir', type=str, help='Output directory.', default='.', required=False)
### sampling parameters ###
parser.add_argument('--alpha', dest='alpha', type=float, help='Alpha.', default=0.1, required=False)
parser.add_argument('--beta_K', dest='beta_K', type=float, help='Beta/K.', default=1.0, required=False)
parser.add_argument('--p_val', dest='p_val', type=float, help='Significance threshold for RRHO method.', default=0.05, required=False)
parser.add_argument('--max_n_steps', dest='max_n_steps', type=int, help='Maximal number of steps.', default=50, required=False)
parser.add_argument('--n_steps_averaged', dest='n_steps_averaged', type=int, help='Number of last steps analyzed when checking convergence condition. Values less than 10 are not recommended.', default=20, required=False)
parser.add_argument('--n_steps_for_convergence', dest='n_steps_for_convergence', type=int, help='Required number of steps when convergence conditions is satisfied.', default=5, required=False)
parser.add_argument('--min_n_samples', dest='min_n_samples', type=int, help='Minimal number of samples on edge.', default=-1, required=False)
### merging and filtering parameters
parser.add_argument('--min_SNR', dest='min_SNR', type=float, help='SNR threshold for biclusters to consider.', default=0.5, required=False)
parser.add_argument('--min_sample_overlap', dest='min_sample_overlap', type=float, help='', default=0.5, required=False)
parser.add_argument('--allowed_SNR_decrease', dest='allowed_SNR_decrease', type=float, help='maximum allowed percent of SNR decrease when merge two modules', default=0.1, required=False)

parser.add_argument('--mem', dest='mem', action='store_true', help='Cache log().', required=False)

### plot flag
parser.add_argument('--plot_all', dest='plot_all', action='store_true', help='Switches on all plotting.', required=False)
### if verbose 
parser.add_argument('--verbose', dest='verbose', action='store_true', help='', required=False)

### whether rewrite temporary files
parser.add_argument('--force', dest='force', action='store_true', help='', default=False, required=False)

########################## Step 1. Read and check inputs ###############################################
start_time = time.time()

args = parser.parse_args()

if args.verbose:
    print("NetworkX version:",nx.__version__, "; must be < 2.", file = sys.stdout)
    
if not args.direction in ["UP","DOWN"]:
    print("Direction of dysregulatoin must be 'UP' or 'DOWN'.", file = sys.stderr)
    exit(1)
    
if not args.method in ["RRHO","top_half"]:
    print("Method must be 'RRHO' or 'top_half'.", file = sys.stderr)
    exit(1) 
    
if args.verbose:
    print("Expression:",args.exprs_file, 
          "\nNetwork:",args.network_file,
          "\n",file = sys.stdout)
    print("alpha:",args.alpha, 
          "\nbeta/K:",args.beta_K,
          "\ndirection:",args.direction,
          "\nmethod:",args.method,
          "\nRRHO significance threshold:",args.p_val,
          "\nmax_n_steps:",args.max_n_steps,
          "\nn_steps_averaged:",args.n_steps_averaged,
          "\nn_steps_for_convergence:",args.n_steps_for_convergence,
          "\nmin_SNR:",args.min_SNR,
          "\nallowed_SNR_decrease:",args.allowed_SNR_decrease,
          "\nmin_sample_overlap:",args.min_sample_overlap,
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

suffix  = ".alpha="+str(args.alpha)+",beta_K="+str(args.beta_K)+","+args.method+"_"+args.direction+",p_val="+str(args.p_val)
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


if args.mem:
    ### log with cache
    def memoize(function):
        cache = {}
        def decorated_function(*args):
            if args in cache:
                return cache[args]
            else:
                val = function(*args)
                cache[args] = val
                return val
        return decorated_function



    my_log = memoize(math.log)
    my_log = np.vectorize(my_log)
else:
    my_log = np.log


### try loading data for initial state
ini_state_file = args.out_dir+args.basename+suffix+".initial_state"
if os.path.exists(ini_state_file+".network.txt") and os.path.exists(ini_state_file+".pickle") :
    
    from desmond_io import load_object, load_network
    
    network = load_network(ini_state_file+".network.txt")
    [moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs] = load_object(ini_state_file+".pickle")
    print("Loaded initial state data from",ini_state_file,file = sys.stdout)
    
    print("\tgenes:",len(network.nodes()),"\tsamples:",N,
          "\n\tnon-empty edges:",len(network.edges()),file = sys.stdout)

else:
    ########################## RRHO ######################
    ### perform RRHO for every edge 
    # first check if the network already exists
    network_with_samples_file =  args.out_dir+ basename +"."+args.method+"_"+args.direction+",p_val="+str(args.p_val)+".network.txt"
    if os.path.exists(network_with_samples_file):
        from desmond_io import print_network_stats,load_network
        network = load_network(network_with_samples_file, verbose = args.verbose)
        print("Loaded annotated network from",network_with_samples_file,file = sys.stdout)
        print_network_stats(network)
    else:
        from method import precompute_RRHO_thresholds,  expression_profiles2nodes, assign_patients2edges, mask_empty_edges
        from desmond_io import save_network
        ##### assign expression vectors on nodes 
        network = expression_profiles2nodes(network, exprs, args.direction)
        
        if args.method == "RRHO":
            # define step for RRHO
            fixed_step = int(max(1,0.01*len(exprs.columns.values))) # 5-10-20 ~15
            if args. verbose:
                print("Fixed step for RRHO selected:", fixed_step, file =sys.stdout)
            rrho_thresholds = precompute_RRHO_thresholds(exprs, fixed_step = fixed_step,significance_thr=args.p_val)
            
        ####  assign patients on edges
        network = assign_patients2edges(network, method= args.method,
                                        fixed_step=fixed_step, rrho_thrs = rrho_thresholds,
                                        verbose=args.verbose)

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
    
###################################### Step 2. Sample module memberships ######################

    ##### set initial model state #######
    from method import set_initial_conditions, calc_lp
    from desmond_io import save_object, save_network
    
    if args.verbose:
        print("Compute initial conditions...",file = sys.stdout)           
    
    moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs = set_initial_conditions(network,exprs ,verbose = args.verbose)

    ### setting the initial state
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
                    lp = calc_lp(e,m,m2,edge2Patients,nOnesPerPatientInModules,moduleSizes,
                                 moduleOneFreqs,p0, match_score,mismatch_score,
                                 bK_1,log_func=my_log,alpha=args.alpha,beta_K=args.beta_K)
                    data['log_p'].append(lp)
                    data['modules'].append(m2)
        if args.verbose and e%1000 == 0:
            print(e,"\tedges processed",round(time.time()- t_1,1) , "s runtime",file=sys.stdout)
            t_1 = time.time()
    print("Set initial LPs in",round(time.time()- t_0,1) , "s", file = sys.stdout)

    ### save data for initial state
    #save_network(network,ini_state_file + ".network.txt" )
    #save_object([moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs], ini_state_file +".pickle" )

edge2Module_history_file = args.out_dir+args.basename+suffix+",ns_max="+str(args.max_n_steps)+",ns_avg="+str(args.n_steps_averaged)+",ns_c="+str(args.n_steps_for_convergence)+".e2m_history.pickle"

if os.path.exists(edge2Module_history_file) and not args.force:
    [edge2Module_history,n_final_steps,n_skipping_edges,P_diffs] = load_object(edge2Module_history_file)
    print("Loaded precomputed model states from",edge2Module_history_file,file = sys.stdout)
    print("Will build consensus of the last",n_final_steps,"states.",file = sys.stdout)
else:
    ### Sampling
    from method import sampling# check_convergence, check_convergence_conditions, apply_changes
    edge2Module_history,n_final_steps,n_skipping_edges,P_diffs = sampling(network,exprs, edge2Module, edge2Patients, nOnesPerPatientInModules,moduleSizes,moduleOneFreqs, p0, match_score,mismatch_score, bK_1,log_func=my_log, alpha = args.alpha, beta_K = args.beta_K, max_n_steps=args.max_n_steps, n_steps_averaged = args.n_steps_averaged, n_points_fit = 10, tol = 0.1, n_steps_for_convergence = args.n_steps_for_convergence, edge_ordering = "corr", verbose=args.verbose)
    # save full edge2Module history
    #save_object([edge2Module_history,n_final_steps,n_skipping_edges,P_diffs],edge2Module_history_file)


if args.plot_all:
    from desmond_io import plot_convergence
    plot_outfile = args.out_dir + args.basename +suffix+",ns_max=" + str(args.max_n_steps)+ ",ns_avg=" + str(args.n_steps_averaged) + ",ns_c="+str(args.n_steps_for_convergence) + ".convergence.svg"
    plot_convergence(n_skipping_edges, P_diffs,len(edge2Module_history)-n_final_steps,
                     args.n_steps_averaged, outfile=plot_outfile)
### take the last (n_points_fit+n_steps_for_convergence) steps
edge2Module_history = edge2Module_history[-n_final_steps:]

### get consensus edge-to-module membership
consensus_edge2module, network, edge2Patients, nOnesPerPatientInModules, moduleSizes, moduleOneFreqs = get_consensus_modules(edge2Module_history, network, edge2Patients, edge2Module, nOnesPerPatientInModules,moduleSizes, moduleOneFreqs, p0,match_score,mismatch_score, bK_1,log_func=my_log, alpha=args.alpha,beta_K=args.beta_K)

print("Empty modules:", len([x for x in moduleSizes if x == 0]),
      "\nNon-empty modules:",len([x for x in moduleSizes if x != 0]))

################################## 3. Post-processing ####################################################
### Make biclusters 
from method import get_genes, identify_opt_sample_set, bicluster_avg_SNR
from desmond_io import write_modules
from method import merge_modules, calc_new_SNR, add_bic, remove_bic
#from sklearn.cluster import KMeans

t0 = time.time()
bics = []
for mid in range(0,len(moduleSizes)):
    if moduleSizes[mid]>2:
        genes = get_genes(mid,edge2Module,network.edges())
        samples, thr, avgSNR = identify_opt_sample_set(nOnesPerPatientInModules[mid,],
                                            exprs, genes, min_n_samples=args.min_n_samples)
        #e = exprs.loc[genes,:]
        #labels = KMeans(n_clusters=2, random_state=0).fit(e.T).labels_
        #if sum(labels) > N/2:
        #    ndx = np.where(labels == 0)[0]
        #else:
        #    ndx = np.where(labels == 1)[0]
        #samples = e.columns[ndx]
        #avgSNR = bicluster_avg_SNR(exprs,genes=genes,samples=samples)
        bics.append({"genes":set(genes), "samples":set(samples), "avgSNR":avgSNR,"id":mid})
print("Identified optimal patient sets in %s",round(time.time()-t0,2))

print("Biclusters with > 2 genes:",len(bics), file = sys.stdout)

####  throw out biclusters with low avg. SNR 
filtered_bics = []
for bic in bics:
    if bic["avgSNR"] > args.min_SNR:
        filtered_bics.append(bic)
        print(len(bic["genes"]),len(bic["samples"]),bic["avgSNR"])
        
print("biclusters after SNR filtering:",len(filtered_bics), file=sys.stdout)

#### Merge remaining biclusters 
"""resulting_bics = merge_modules(filtered_bics,nOnesPerPatientInModules,moduleSizes,exprs,
                               min_sample_overlap = args.min_sample_overlap,
                               min_acceptable_SNR_percent=1-args.allowed_SNR_decrease,
                               min_n_samples=args.min_n_samples, verbose= args.verbose)
print("biclusters after merge:",len(resulting_bics))
"""
resulting_bics = filtered_bics

result_file_name = args.out_dir+args.basename+suffix+",ns_max="+str(args.max_n_steps)+",ns_avg="+str(args.n_steps_averaged)+",ns_c="+str(args.n_steps_for_convergence)+".biclusters"

if args.plot_all:
    from desmond_io import plot_bic_stats
    plot_outfile = args.out_dir + args.basename + suffix+",ns_max="+str(args.max_n_steps)+",ns_avg="+str(args.n_steps_averaged)+",ns_c="+str(args.n_steps_for_convergence)+".bicluster_stats.svg"
    plot_bic_stats(bics,plot_outfile)

write_modules(resulting_bics, result_file_name+".txt")
print("Total runtime:",round(time.time()-start_time,2),file = sys.stdout)

resulting_bics = pd.DataFrame.from_dict(resulting_bics)
resulting_bics["n_genes"] = resulting_bics["genes"].apply(len)
resulting_bics["n_samples"] =  resulting_bics["samples"].apply(len)
resulting_bics["genes"] = resulting_bics["genes"].apply(lambda x:" ".join(map(str,x)))
resulting_bics["samples"] = resulting_bics["samples"].apply(lambda x:" ".join(map(str,x)))
resulting_bics = resulting_bics[["id","avgSNR","n_genes","n_samples","genes","samples"]]
resulting_bics.sort_values(by=["n_samples","avgSNR","n_genes"],inplace = True, ascending = False)
resulting_bics.to_csv(result_file_name+".tsv" ,sep = "\t")
