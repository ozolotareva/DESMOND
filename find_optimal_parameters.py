from __future__ import print_function
import pandas as pd
import numpy as np
from itertools import product
import sys,os
import time

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

def read_true_bics(bic_file_path):
    with open(bic_file_path,"r") as infile:
        bics = []
        for line in infile.readlines():
            line = line.rstrip().split()
            if line[0].startswith("bic"):
                bic = {"id":line[1]}
            elif line[0].startswith("g"):
                bic["genes"] = set(line[1:])
            else:
                bic["samples"] = set(line[1:])
                bics.append(bic)
    return bics



def read_all_results(parameters,tool_name,n_runs=10,
                     pred_bic_dir="./",pred_bic_fname_prefix="simulated.N=10.Mu=2.0.GxP=",
                     pred_bic_fname_suffix=".biclust_results.txt",
                     true_bic_dir="./",true_bic_fname_prefix="simulated.N=10.Mu=2.0.GxP=",
                     true_bic_fname_suffix=".biclusters.txt",
                     g_sizes = [5,10,20,50,100], s_sizes = [10,20,50,100]):
    results = []

    failed_param_combinations = 0
    succ_param_combinations = 0
    files_not_found = 0
    files_empty = 0
    
    true_biclusters = {} # {(n_genes,n_samples):true_bics}
    for n_genes in g_sizes:
        for n_samples in s_sizes:
            #read_true_bics
            true_bic_fname = true_bic_fname_prefix+str(n_genes)+","+str(n_samples)+true_bic_fname_suffix 
            true_bics = read_true_bics(true_bic_dir+true_bic_fname)
            true_biclusters[(n_genes,n_samples)] =  true_bics
    # generate parameter combinations:
    params = []
    product_target = []
    for param in parameters:
        params.append(param[0])
        product_target.append(param[1])

    
    for param_combination in product(*product_target):
        param_combination = zip(params,param_combination)
        params_folder = ".".join(map(lambda x: x[0]+"="+str(x[1]),param_combination))
    
        if not os.path.exists(pred_bic_dir+params_folder+"/"):
            print(pred_bic_dir+params_folder+"/","does not found.")
            failed_param_combinations += 1
        else:
            succ_param_combinations += 1
            for n_genes in g_sizes:
                for n_samples in s_sizes:
                    pred_bic_fname = pred_bic_fname_prefix+str(n_genes)+","+str(n_samples)+pred_bic_fname_suffix
                    if not os.path.isfile(pred_bic_dir+params_folder+"/" + pred_bic_fname):
                        files_not_found+=1
                        runs = False
                    elif os.path.getsize(pred_bic_dir+params_folder+"/" + pred_bic_fname)== 0:
                        files_empty +=1
                        runs = False
                    else:
                        try:
                            runs = parse_biclust(pred_bic_dir+params_folder+"/" + pred_bic_fname, n_runs = n_runs)
                        except:
                            runs = False
                            print("Failed to parse",pred_bic_dir+params_folder+"/" + pred_bic_fname)
                    if not runs:
                        for run in range(1,n_runs+1):
                            d = {"n_run":run,"F1-like":0,"n_genes":n_genes,"n_samples":n_samples}
                            d.update(dict(param_combination))
                            results.append(d)
                    else:
                        n_run = 1
                        for pred_bics in runs:
                            F1 = compute_F1like(true_biclusters[(n_genes,n_samples)], pred_bics, verbose = False)
                            d = {"n_run":n_run,"n_genes":n_genes,"n_samples":n_samples,"F1-like":np.mean(F1)}
                            d.update(dict(param_combination)) # {"discr_levels":discr_levels,"alpha":alpha, "ns":ns,"n_run":n_run,
                            results.append(d)
                            n_run+=1



    print("Parameter combiations not found: %s" % (failed_param_combinations))
    print("Successfull parameter combinations: %s" % (succ_param_combinations))
    print("\tfiles not found: %s" % (files_not_found))
    print("\tfiles empty: %s" % (files_empty))
    results=pd.DataFrame.from_dict(results)
    #results=results.loc[results["alpha"]==0.05,:].loc[results["ns"]==10,:]
    print("Total runs",results.shape[0])
    print("Non-zero runs:",results.loc[results["F1-like"]>0,:].shape[0])
    return results 

def parse_biclust(bic_file_path, n_runs = 10):
    runs = []
    bics = []
    prev_n_run = 0
    with open(bic_file_path,"r") as infile:
        for line in infile.readlines():
            if line.startswith("Sample_id"):
                if len(bics) > 0:
                    # close prev bics
                    runs.append(bics)
                # open new bics
                bics = []
                i = 0
                bic_id = 1
                
                n_run = line.split("; ")[-1]
                if n_runs > 1:
                    n_run = int(n_run.replace("Run:",""))
                    # add empty runs to runs[] if necessary
                    for run in range(prev_n_run+1,n_run):
                        runs.append([{"id":bic_id ,"run":run,"genes":set([]),"samples":set([]),
                                      "n_genes":0,"n_samples":0}])
                        bic_id +=1
                    prev_n_run =  n_run 
                        
                else:
                    n_run = 1
                
            else:
                line = line.rstrip().split()
                if i%3 == 0:
                    # open bic
                    n_genes, n_samples = int(line[0]), int(line[1])
                    bic = {"id":bic_id,"run":n_run,"n_genes":n_genes, "n_samples":n_samples}
                    bic_id +=1
                    i+=1
                elif i%3 == 1:
                    if bic["n_genes"]>0:
                        bic["genes"] = set(line)
                        if bic["n_samples"] == 0:
                            bic["samples"]=set([])
                            bics.append(bic)
                            i=0
                        else:
                            i+=1
                    else:
                        bic["genes"] = set([])
                        if bic["n_samples"]>0:
                            if len(line) > 0:
                                bic["samples"]=set(line)
                                bics.append(bic)
                                i=0
                            else:
                                i+=1
                        else:
                            bic["samples"]=set([])
                            bics.append(bic)
                            if len(line) > 0:
                                i=0
                            else:
                                i+=1
                else:
                    bic["samples"]=set(line)
                    bics.append(bic)
                    i=0
    runs.append(bics)
    if n_runs > 1:
        for run in range(prev_n_run+1,n_runs+1):
            runs.append([{"id":bic_id ,"run":run,"genes":set([]),"samples":set([]),
                                          "n_genes":0,"n_samples":0}])
            bic_id +=1
    if len(runs)!=n_runs:
        print("Warining:  %s runs found in file %s instead of %s expected" %(len(runs),bic_file_path,n_runs ))
    return runs

def compute_F1like(true_bics, pred_bics, verbose = False):
    if verbose:
        print("\t".join(["id","best_match_id","%bic_recovered",
                        "true_genes","true_samples", "f1"]))
    if len(pred_bics) == 0 :
        return list(np.zeros(len(true_bics)))
    F1 = []
    for t in true_bics:
        tg = t["genes"]
        ts = t["samples"]
        bm_id = False
        best_overlap = 0
        for bic in pred_bics:
            n_shared_genes = len(tg.intersection(bic["genes"]))
            n_shared_samples = len(ts.intersection(bic["samples"]))
            overlap = n_shared_genes * n_shared_samples
            if overlap > best_overlap:
                best_overlap, shared_genes, shared_samples = overlap, n_shared_genes, n_shared_samples
                bm_id = bic["id"]
                bm_genes = len(bic["genes"])
                bm_samples = len(bic["samples"])
        if bm_id:
            tp = best_overlap 
            fp = bm_genes*bm_samples-tp
            fn = len(t["genes"])*len(t["samples"]) -tp
            prec = 1.0*tp/(tp + fp) 
            rec =  1.0*tp/(tp + fn)
            f1 = 2*(prec*rec)/(prec+rec)
        else:
            f1=0
        F1.append(f1)
        if verbose:
            print("\t".join(map(str,[t["id"],bm_id,
                                 1.0*best_overlap/(len(t["genes"])*len(t["samples"])),
                                 shared_genes, shared_samples, f1])))
    return F1


def plot_F1_heatmap(results,params,f1_thr=0.05,g_sizes = [5,10,20,50,100], s_sizes = [10,20,50,100],
                    plot=True,plot_file="",outfile=""):
    
    print("Total combinations:", results.loc[:,params].drop_duplicates().shape[0])
    heatmap = {}
    for n_genes in g_sizes:
        for n_samples in s_sizes:
            r = results.loc[results["n_genes"]==n_genes,:].loc[results["n_samples"]==n_samples,:]
            r = r.groupby(by=params).agg(np.mean)[["F1-like"]]
            heatmap[(n_genes ,n_samples)] = r.to_dict()['F1-like']
    heatmap = pd.DataFrame.from_dict(heatmap)
    heatmap.fillna(0,inplace = True)
    heatmap.index.names = params
    heatmap.columns.names = ["genes","samples"]
    print("Total combinations:", heatmap.shape[0])
    heatmap_show = heatmap.loc[heatmap.apply(np.mean,axis=1)  > f1_thr,:]
    print("Combinations with mean F1 > "+str(f1_thr),heatmap_show.shape[0])
    if plot:
        fig, ax = plt.subplots(figsize=(20,10*heatmap_show.shape[0]/40+1))
        sns.heatmap(heatmap_show,ax=ax, annot=True)
        if plot_file:
            plt.savefig(plot_file)
    if outfile:
        heatmap.to_csv(outfile,sep="\t")
    return heatmap 


def get_opt_params(results, params, more_n_smaples = 0, default_params = None, verbose=True):
    if more_n_smaples > 0:
        ini_runs  = results.shape[0]
        r = results.loc[results["n_samples"]> more_n_smaples,:]
        print("%s runs considered, %s runs on examples of biclusters with < %s samples excluded" % (r.shape[0], ini_runs - r.shape[0],more_n_smaples+1))
    else:
        r = results
        print("All %s runs considered" % (r.shape[0])) 
        
    n_bic_kinds = r[["n_genes","n_samples"]].drop_duplicates().shape[0]
    r = r.groupby(by = params).agg(["mean","std","count","max","min"])[["F1-like"]]
    r[("F1-like","n_runs")] =  r[("F1-like","count")]*1.0/n_bic_kinds
    r = r.sort_values(by = ("F1-like","mean"), ascending= False)
    if verbose:
        m = round(r.head(1).loc[:,("F1-like","mean")].values[0],3)
        #err = round(r.head(1).loc[:,("F1-like","std")].values[0]/np.sqrt(r.head(1).loc[:,("F1-like","count")].values[0]),3) # std/sqrt(n)
        std = round(r.head(1).loc[:,("F1-like","std")].values[0],3)
        print("Max. avg. F1:"+str(m)+u"\u00B1"+str(std))
        print("\nparameters:")
        for p in zip(params, r.head(1).index.values[0]):
            print("\t"+p[0]+"="+str(p[1])+";")
        if default_params:
            m_def = round(r.loc[default_params,("F1-like","mean")],3)
            std_def = round(r.loc[default_params,("F1-like","std")],3)
            print("With def. parameters",zip(params,default_params),
                  "avg. F1:"+str(m_def)+u"\u00B1"+str(std_def))
    return r