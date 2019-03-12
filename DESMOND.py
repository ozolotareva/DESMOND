from __future__ import print_function
import sys,os

import numpy as np
from scipy.stats import hypergeom # ,fisher_exact
from fisher import pvalue
import pandas as pd
 
import networkx as nx
import ndex2.client
import ndex2

import itertools
import warnings
import time
import datetime
import copy
import random
import argparse

#sys.path.append(os.path.abspath("/home/olya/SFU/Breast_cancer/DESMOND/"))
from method import prepare_input_data, print_network_stats
from method import precompute_RRHO_thresholds

from method import  expression_profiles2nodes, assign_patients2edges, save_subnetworks,load_subnetworks
from method import plot_patient_ditsribution_and_mask_edges, count_emtpy_edges, count_masked_edges

from method import  set_initial_distribution,sampling, get_consensus_module_membrship
from method import check_convergence_condition, restore_modules, get_genes, get_opt_pat_set
from post_processing import plot_bic_stats, write_modules, merge_modules

parser = argparse.ArgumentParser(description="""Searches for genes differentially expressed in an unknown subgroup of samples.""" , formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-e','--exprs', dest='exprs_file', type=str, help='Expression matrix with sample name in columns and gene in rows, tab-separated.', default='', required=True,metavar='exprs_zscores.tsv')
parser.add_argument('-n','--network', dest='network_file', type=str, help='Network in NDEX2 format.', default='', required=True, metavar='network.cx')
parser.add_argument('-s','--seeds', dest='seeds_file', type=str, help='File with seed gene names.', default='', required=False)
parser.add_argument('-d','--direction', dest='direction', type=str, help='Direction of dysregulation: UP or DOWN', default='UP', required=False)
parser.add_argument('-m','--method', dest='method', type=str, help='How to assign patients on edges: RHHO or top_halves', default='RRHO', required=False)
parser.add_argument('-basename','--basename', dest='basename', type=str, help='Output basename without extention. If no outfile name provided output will be set "results_hh:mm_dddd-mm-yy".', default='', required=False)
parser.add_argument('-o','--out_dir', dest='out_dir', type=str, help='Output directory.', default='.', required=False)
### sampling parameters ###
parser.add_argument('--alpha', dest='alpha', type=float, help='Alpha.', default=0.1, required=False)
parser.add_argument('--beta_K', dest='beta_K', type=float, help='Beta/K.', default=1.0, required=False)
parser.add_argument('--max_n_steps', dest='max_n_steps', type=int, help='Maximal number of steps.', default=100, required=False)
parser.add_argument('--min_pletau_steps', dest='min_pletau_steps', type=int, help='Minimal number of steps satisfying pletau condition.', default=30, required=False)
### RWR parameters ###
parser.add_argument('--r', dest='r', type=float, help='Restart probability. Controlls the expected number of steps, e.g. for r=0.3 is 2.3.', default=0.3, required=False)
parser.add_argument('--delta', dest='delta', type=float, help='RWR tolerance.', default=0.000001, required=False)
parser.add_argument('--rwr_thr', dest='rwr_t', type=float, help='RWR probability threshold. Determines how large subnetworks will be; strongly depends on the network topology.', default=0.001, required=False)
### plot flag
parser.add_argument('--plot_all', dest='plot_all', action='store_true', help='Switches on all plotting.', default=False, required=False)
### if verbose 
parser.add_argument('--verbose', dest='verbose', action='store_true', help='', default=True, required=False)
### whether rewrite temporary files
parser.add_argument('--force', dest='force', action='store_true', help='', default=False, required=False)

########################## Read input paramethers ############
args = parser.parse_args()


if args.verbose:
    print("NetworkX version:",nx.__version__, "; must be < 2.", file = sys.stderr)
    
if not args.direction in ["UP","DOWN"]:
    print("Direction of dysregulatoin must be 'UP' or 'DOWN'.", file = sys.stderr)
    exit(1)
    
if not args.method in ["RRHO","top_half"]:
    print("Method must be 'RRHO' or 'top_half'.", file = sys.stderr)
    exit(1) 

# where to write the results
# create directory if it does not exists 
if not os.path.exists(args.out_dir) and not args.out_dir == "." :
    os.makedirs(args.out_dir)
args.out_dir = args.out_dir + "/"
# set output basename
if args.basename:
    basename = args.basename
else: 
    [date_h,mins] = str(datetime.datetime.today()).split(":")[:2]
    [date, hs] = date_h.split()
    basename = "results_"+hs+":"+mins+"_"+date 
if args.seeds_file:
    basename +="_r"+str(args.r)+"_T"+str(args.rwr_thr)+"."+args.method+"_"+args.direction
else:
    basename +="."+args.method+"_"+args.direction
if args.verbose:
    print("basename:", basename, file = sys.stderr)
    
    
# read data files
# seeds are optional and not used by default
if args.seeds_file: 
    from method import prepare_input_data, print_network_stats
    from method import precompute_RRHO_thresholds
    from RWR import create_subnetworks_from_seeds #calc_distanse_matrix, group_seed_genes
    exprs, network, seeds = prepare_input_data(args.exprs_file, args.network_file, args.seeds_file, verbose = args.verbose)
else: 
    # read and preprocess input files: network and expressions
    exprs, network = prepare_input_data(args.exprs_file, args.network_file, verbose = args.verbose)

if args.verbose:
    print_network_stats(network, print_cc = True)
    
# define minimal number of patients in a module
min_n_patients = int(max(10,0.1*len(exprs.columns.values))) # set to nax(10, 10% of the cohort) 
# print("Fixed step for RRHO selected:", ,file =sys.stderr)
if args. verbose:
    print("Mininal number of patients in a module:",min_n_patients ,file=sys.stderr)
          
### Optional: RWR ###
if  args.seeds_file:
    if args.verbose:
        print("Run RWR to get subnetworks around seed genes",file = sys.stderr)
    network  = create_subnetworks_from_seeds(network,seeds,args.r,args.delta,args.rwr_thr, verbose = args.verbose)
    # make network from a list of subnetworks
    network = nx.compose_all(network)

##### Step 1. Assign patients to edges #####
network_with_pats_file =  args.out_dir+ basename +".network.txt"
if not os.path.exists(network_with_pats_file):
    # assign expression vectors on nodes 
    network = expression_profiles2nodes(network, exprs, args.direction)
    # RRHO - how to assign patients on edges
    if args.method == "RRHO":
        # set RRHO parameters
        significance_thr=0.05
        fixed_step = int(max(1,0.02*len(exprs.columns.values))) # 5-10-20 ~15
        if args.verbose:
            print("Fixed step for RRHO selected:", fixed_step, file =sys.stderr)
        rrho_thresholds = precompute_RRHO_thresholds(exprs, fixed_step = fixed_step,significance_thr=significance_thr)
    network = assign_patients2edges(network, method= args.method,
                                    fixed_step=fixed_step,significance_thr=significance_thr,
                                    rrho_thrs = rrho_thresholds)
    if args.verbose:
        print("Edges without any patients:",count_emtpy_edges(network, thr = 0),file= sys.stderr)


    # save the network with patients on edges 
    save_subnetworks([network], network_with_pats_file)
    if args.verbose:
        print("Write network with patients to",network_with_pats_file,file= sys.stderr)
else:
    network = load_subnetworks(network_with_pats_file, verbose = args.verbose)
    network = nx.compose_all(network)

plot_patient_ditsribution_and_mask_edges(network,min_n_patients=min_n_patients,title="Distribution of the number of patients over edges.", plot = False)
if args.verbose:
    print("edges",len(network.edges()),"edges masked (i.e. edges without patients):",count_masked_edges(network),file= sys.stderr)

### Step 2. Sample module memberships ###
# set initial model state
# initial distribution
if args.verbose:
    print("Set initial conditions...",file = sys.stderr)
edges, edgeNeighorhood, edge2module, moduleSizes,  edge2patients, nOnesPerPatientInModules = set_initial_distribution(network, exprs, basename, args.out_dir)

# sampling 
if args.verbose:
    print("Start sampling until model convergence...",file = sys.stderr)
edge2module_history =  sampling(edges,  edgeNeighorhood, edge2module, edge2patients,  moduleSizes,nOnesPerPatientInModules, max_n_steps=args.max_n_steps,alpha = args.alpha, beta_K = args.beta_K,min_pletau_steps = args.min_pletau_steps)
# save intermidiate results 
np.save(args.out_dir+basename+".edge2module_history.a="+str(args.alpha)+".b="+str(args.beta_K), edge2module_history)
np.save(args.out_dir+basename+".edges.a="+str(args.alpha)+".b="+str(args.beta_K), edges)

# get consensus
# consensus edge-to-module membership
if args.verbose:
    print("Get consensus of edge memberships...",file = sys.stderr)
consensus_edge2module = get_consensus_module_membrship(edge2module_history, edges)

# make moduleSizes, nOnesPerPatientInModules corresponding to consensus module mebership 
moduleSizes, nOnesPerPatientInModules = restore_modules(consensus_edge2module,edges,network,exprs)

print("empty modules:", len([x for x in moduleSizes if x == 0]),
      "non-empty modules:",len([x for x in moduleSizes if x != 0]), file = sys.stderr)
if args.plot_all:
    tmp = plt.hist(moduleSizes, bins=50, range=(1,max(moduleSizes)))

# turn modules to biclusters 
bics = []
for mid in range(0,len(moduleSizes)):
    if moduleSizes[mid]>1:
        genes = get_genes(mid,edge2module=consensus_edge2module,edges=edges)
        pats, thr, avgSNR = get_opt_pat_set(nOnesPerPatientInModules[mid,], moduleSizes[mid,],
                                            exprs, genes, min_n_patients=min_n_patients)
        bics.append({"genes":set(genes), "patients":set(pats), "avgSNR":avgSNR,"id":mid})

if args.plot_all:
    plot_bic_stats(bics)

#####  Step 3. Filter and merge biclusters ####
min_SNR = 0.5
min_patient_overlap = 0.75 # at least 75 % of candidates 
allowed_SNR_decrease  = 0.1 # maximum allowed % of SNR decrease when merge two modules 
# -i.e. new module loses after merge no more than 10% of SNR 

filtered_bics = []
for bic in bics:
    if bic["avgSNR"] > min_SNR:
        filtered_bics.append(copy.copy(bic))
plot_bic_stats(filtered_bics)

resulting_modules = merge_modules(filtered_bics, nOnesPerPatientInModules,moduleSizes,exprs,SNRs = [],
                              min_patient_overlap = min_patient_overlap,
                              min_acceptable_SNR_percent=1-allowed_SNR_decrease, verbose= False)

# write to file
file_name = args.out_dir+"/"+basename+".minSNR_"+str(min_SNR)+".modules.txt"
write_modules(resulting_modules,file_name)