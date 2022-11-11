import sys
import copy
import random
import pandas as pd
import numpy as np
import networkx as nx
import time
import math
import itertools
import os

from sklearn.cluster import KMeans
from fisher import pvalue
import matplotlib.pyplot as plt


def prepare_input_data(exprs_file, network_file, verbose=True, min_n_nodes=3):
    '''1) Reads network in .tab format. Converts the network into undirected, renames nodes.
    \n2) Keeps only genes presenting in the network and expression matrix and retains only large connected components with more than min_n_nodes nodes. '''
    t0 = time.time()
    # read expressoin matrix
    exprs = pd.read_csv(exprs_file, sep="\t", index_col=0)
    # exprs.rename(str,axis="index",inplace=True) # this is because
    exprs_genes = exprs.index.values

    if len(set(exprs_genes)) != len(exprs_genes):
        if verbose:
            print("Duplicated gene names", file=sys.stderr)

    # read and prepare the network
    # read  undirected network .tab network, e.g.gene1 gene2 ...
    network_df = pd.read_csv(network_file, sep="\t", header=None)
    if network_df.shape[1] < 2:
        network_df = pd.read_csv(network_file, sep=" ", header=None)
    network = nx.from_pandas_edgelist(network_df, source=0, target=1)

    network_genes = network.nodes()
    ccs = list(nx.connected_components(network))
    if verbose:
        print("Input:\n", "\texpressions:", len(exprs_genes), "genes x", len(set(exprs.columns.values)), "samples;",
              "\n\tnetwork:", len(network_genes), "genes,", len(network.edges()), "edges in", len(ccs), "connected components:", file=sys.stdout)

    network_genes = network.nodes()

    #### compare genes in network and expression and prune if necessary  ###
    genes = set(network_genes).intersection(set(exprs_genes))
    if len(genes) != len(network_genes):
        # exclude unnecessary genes from the network
        network_edges = len(network.edges())
        network = nx.subgraph(network, genes)
        if verbose:
            print(len(network_genes)-len(genes), "network nodes without expressoin profiles and",
                  network_edges-len(network.edges()), "edges excluded", file=sys.stdout)
    if len(genes) != len(exprs_genes):
        # exclude unnecessary genes from the expression matrix
        exprs = exprs.loc[genes, :]
        if verbose:
            print(len(exprs_genes)-len(genes),
                  "genes absent in the network excluded from the expression matrix", file=sys.stdout)

    # remove small CCs containing less than min_n_nodes
    network = remove_small_cc(network, min_n_nodes=2)
    network_genes = list(network.nodes())
    exprs = exprs.loc[network_genes, :]
    ccs = list(nx.connected_components(network))

    if verbose:
        print("Processed Input:\n", "\texpressions:", len(exprs.index.values), "genes x", len(set(exprs.columns.values)), "samples;",
              "\n\tnetwork:", len(network_genes), "genes ", len(network.edges()), "edges in", len(ccs), "connected components:", file=sys.stdout)
        print("time:\tInputs read in %s s." %
              round(time.time()-t0, 2), file=sys.stdout)
    return exprs, network


def remove_small_cc(network, min_n_nodes=2):
    # remove small CCs containing less than min_n_nodes
    nodes = []
    ccs = list(nx.connected_components(network))  # sets of nodes
    for cc in ccs:
        if len(cc) >= min_n_nodes:
            nodes += cc
    network = nx.subgraph(network, nodes)
    return network


def print_network_stats(G, print_cc=True):
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    if nx.is_directed(G):
        is_directed = "Directed"
    else:
        is_directed = "Undirected"
    print(is_directed, "graph with", len(ccs), "connected components; with", len(
        G.nodes()), "nodes and", len(G.edges()), "edges;", file=sys.stdout)
    if print_cc and len(ccs) > 1:
        i = 0
        for cc in ccs:
            i += 1
            print("Connected component", i, ":", len(cc), "nodes and", len(
                nx.subgraph(G, cc).edges()), "edges", file=sys.stdout)


def identify_opt_sample_set(genes, exprs, exprs_data, direction="UP", min_n_samples=8):
    # identify optimal samples set given gene set
    N, exprs_sums, exprs_sq_sums = exprs_data
    e = exprs[genes, :]

    labels = KMeans(n_clusters=2, random_state=0).fit(e.T).labels_
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1), len(ndx0)) < min_n_samples:
        return {"avgSNR": -1}
    if np.mean(e[:, ndx1].mean()) > np.mean(e[:, ndx0].mean()):
        if direction == "UP":
            samples = ndx1
        else:
            samples = ndx0
    else:
        if direction == "UP":
            samples = ndx0
        else:
            samples = ndx1
    avgSNR = calc_bic_SNR(genes, samples, exprs, N, exprs_sums, exprs_sq_sums)

    # allow bicluster to be a little bit bigger than N/2
    if len(samples) < N*0.5*1.1 and len(samples) >= min_n_samples:
        bic = {"genes": set(genes), "n_genes": len(genes),
               "samples": set(samples), "n_samples": len(samples),
               "avgSNR": avgSNR, "direction": direction}
        return bic
    else:
        return {"avgSNR": -1}


def calc_bic_SNR(genes, samples, exprs, N, exprs_sums, exprs_sq_sums):
    bic = exprs[genes, :][:, samples]
    bic_sums = bic.sum(axis=1)
    bic_sq_sums = np.square(bic).sum(axis=1)

    bg_counts = N - len(samples)
    bg_sums = exprs_sums[genes]-bic_sums
    bg_sq_sums = exprs_sq_sums[genes]-bic_sq_sums

    bic_mean, bic_std = calc_mean_std_by_powers(
        (len(samples), bic_sums, bic_sq_sums))
    bg_mean, bg_std = calc_mean_std_by_powers((bg_counts, bg_sums, bg_sq_sums))

    return np.mean(abs(bic_mean - bg_mean) / (bic_std + bg_std))


def calc_mean_std_by_powers(powers):
    count, val_sum, sum_sq = powers

    if count == 0:
        return np.nan, np.nan

    mean = val_sum / count
    std = np.sqrt((sum_sq / count) - mean*mean)
    return mean, std

############### save and load temporary results #######################


def load_network(infile_name, verbose=True):
    t0 = time.time()
    '''Reads subnetworks from file.'''
    # read from file
    print("Loading annotated network from ",
          infile_name, "...", file=sys.stdout)
    network = nx.read_edgelist(infile_name)
    for n1, n2 in network.edges():
        if "samples" in network[n1][n2].keys():
            network[n1][n2]["samples"] = set(network[n1][n2]["samples"])
    try:
        network = nx.relabel_nodes(network, int)
    except:
        pass
    if verbose:
        print("time: load_network() runtime", round(
            time.time()-t0, 2), "s", file=sys.stdout)
    return network


def save_network(network, outfile_name, verbose=True):
    t0 = time.time()
    '''Writes subnetwork with associated samples on edges.'''
    # make graph of n subnetworks
    to_save = []
    # modify samples: set -> list
    for n1, n2 in network.edges():
        if "samples" in network[n1][n2].keys():
            network[n1][n2]["samples"] = list(network[n1][n2]["samples"])
    # save to file
    nx.write_edgelist(network, outfile_name, data=True)
    if verbose:
        print("time: save_network() runtime", round(
            time.time()-t0, 2), "s", file=sys.stdout)


################## 2. Probabiliatic clustering #############
def calc_lp(edge, self_module, module, edge2Patients,
            nOnesPerPatientInModules, moduleSizes,
            moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
            alpha=1.0, beta_K=1.0):

    N = edge2Patients.shape[1]
    m_size = moduleSizes[module]
    edge_vector = edge2Patients[edge, ]

    if m_size == 0:
        return p0

    n_ones_per_pat = nOnesPerPatientInModules[module, ]

    if self_module == module:  # remove the edge from the module, if it belongs to a module
        if m_size == 1:
            return p0
        m_size -= 1
        n_ones_per_pat = n_ones_per_pat-edge_vector

    # if a module is composed of a single edge
    if m_size == 1:
        # just count number of matches and mismatches and
        n_matches = np.inner(n_ones_per_pat, edge_vector)
        return n_matches*match_score+(N-n_matches)*mismatch_score + bK_1

    # if a module contains more than one edge
    beta_term = math.log(m_size+beta_K)

    # alpha_term
    # ones-matching
    oneRatios = (n_ones_per_pat+alpha/2)/(m_size+alpha)
    ones_matching_term = np.inner(np.log(oneRatios), edge_vector)
    # zero-matching
    zeroRatios = (m_size-n_ones_per_pat+alpha/2)/(m_size+alpha)
    zeros_matching_term = np.inner(np.log(zeroRatios), (1-edge_vector))

    return ones_matching_term+zeros_matching_term + beta_term


def set_initial_conditions(network, p0, match_score, mismatch_score, bK_1, N, alpha=1.0,
                           beta_K=1.0, verbose=True):
    if verbose:
        print("Compute initial conditions ...", file=sys.stdout)
    t_0 = time.time()

    # 1. the number of edges inside each component, initially 1 for each component
    moduleSizes = np.ones(len(network.edges()), dtype=np.int)

    # 2. a binary (int) matrix of size n by m that indicates the samples on the edges
    edge2Patients = []
    all_samples = range(N)
    for edge in network.edges():
        n1, n2 = edge
        samples_in_module = network[n1][n2]["samples"]
        x = np.zeros(len(all_samples), dtype=np.int)
        i = 0
        for p in all_samples:
            if p in samples_in_module:
                x[i] = 1
            i += 1
        edge2Patients.append(x)

    edge2Patients = np.asarray(edge2Patients)

    # 3. a binary matrix of size K by m that stores the total number of ones per patient in each module,
    # initially equal to 'edge2Patients'
    nOnesPerPatientInModules = copy.copy(edge2Patients)

    t_0 = time.time()
    i = 0
    for n1, n2, data in network.edges(data=True):
        # del data['masked']
        del data['samples']
        data['m'] = i
        data['e'] = i
        i += 1
    # 4.
    edge2Module = list(range(0, len(network.edges())))

    # 5. moduleOneFreqs
    moduleOneFreqs = []
    n = edge2Patients.shape[0]
    for e in range(0, n):
        moduleOneFreqs.append(
            float(sum(edge2Patients[e, ]))/edge2Patients.shape[1])

    # 6. setting initial LPs
    t_0 = time.time()
    t_1 = t_0
    for n1, n2, data in network.edges(data=True):
        m = data['m']
        e = data['e']
        data['log_p'] = []
        data['modules'] = []
        for n in [n1, n2]:
            for n3 in network[n].keys():
                m2 = network[n][n3]['m']
                if not m2 in data['modules']:
                    lp = calc_lp(e, m, m2, edge2Patients,
                                 nOnesPerPatientInModules, moduleSizes,
                                 moduleOneFreqs, p0, match_score, mismatch_score,
                                 bK_1, alpha=alpha, beta_K=beta_K)
                    data['log_p'].append(lp)
                    data['modules'].append(m2)
    print("time:\tInitial state created in", round(
        time.time() - t_0, 1), "s.", file=sys.stdout)
    return moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs, network


def adjust_lp(log_probs, n_exp_orders=7):
    # adjusting the log values before normalization to avoid under-flow
    max_p = max(log_probs)
    probs = []
    for lp in log_probs:
        # shift all probs to set max_prob less than log(max_np.float)
        adj_lp = lp - max_p
        # set to minimal values all which less then 'n_orders' lesser than p_max to zeroes
        if adj_lp >= - n_exp_orders:
            probs.append(np.exp(adj_lp))
        else:
            probs.append(0)
    probs = probs/sum(probs)
    return probs

### functions for checking of convergence conditions ###


def calc_p_transitions(states, unique, counts):
    n_steps = len(states)-1
    transitions = dict(
        zip(tuple(itertools.product(unique, unique)), np.zeros(len(unique)**2)))
    for i in range(0, n_steps):
        transitions[(states[i], states[i+1])] += 1
    p = {k: v/(counts[unique.index(k[0])]) for k, v in transitions.items()}
    return p


def collect_all_p(labels):
    P = {}
    # calculate edge transition probabilities
    for edge in range(0, labels.shape[1]):
        states = labels[:, edge]
        unique, counts = np.unique(states, return_counts=True)
        if len(unique) > 1:
            P[edge] = calc_p_transitions(states, list(unique), counts)
    return P


def calc_RMSD(P, P_prev):
    t0 = time.time()
    p_prev_edges = set(P_prev.keys())
    p_edges = set(P.keys())
    Pdiff = []
    for edge in p_edges.difference(p_prev_edges):
        P_prev[edge] = {k: 0 for k in P[edge].keys()}
        P_prev[edge] = {k: 1 for k in P_prev[edge].keys() if k[0] == k[1]}
    for edge in p_prev_edges.difference(p_edges):
        P[edge] = {k: 0 for k in P_prev[edge].keys()}
        P[edge] = {k: 1 for k in P[edge].keys() if k[0] == k[1]}
    for edge in p_edges.intersection(p_prev_edges):
        p_modules = set(P[edge].keys())
        p_prev_modules = set(P_prev[edge].keys())
        for m, m2 in p_modules.difference(p_prev_modules):
            Pdiff.append((P[edge][(m, m2)])**2)
        for m, m2 in p_prev_modules.difference(p_modules):
            Pdiff.append((P_prev[edge][(m, m2)])**2)
        for m, m2 in p_modules.intersection(p_prev_modules):
            Pdiff.append((P[edge][(m, m2)] - P_prev[edge][(m, m2)])**2)
    if not len(Pdiff) == 0:
        return np.sqrt(sum(Pdiff)/len(Pdiff))
    else:
        return 0


def check_convergence_conditions(n_skipping_edges, n_skipping_edges_range,
                                 P_diffs, P_diffs_range, step, tol=0.05, verbose=True):
    n_points = len(n_skipping_edges)
    # check skipping edges
    se_min, se_max = n_skipping_edges_range
    n_skipping_edges = np.array(n_skipping_edges, dtype=float)

    # scale
    n_skipping_edges = (n_skipping_edges-se_min)/(se_max - se_min)*n_points
    # fit line

    A = np.vstack([range(0, n_points), np.ones(n_points)]).T
    k, b = np.linalg.lstsq(A, n_skipping_edges, rcond=None)[0]

    # check P_diffs
    P_diffs_min, P_diffs_max = P_diffs_range
    P_diffs = np.array(P_diffs)

    # scale
    P_diffs = (P_diffs-P_diffs_min)/(P_diffs_max - P_diffs_min)*n_points
    k2, b2 = np.linalg.lstsq(A, P_diffs, rcond=None)[0]
    if abs(k) < tol and abs(k2) < tol:
        convergence = True
    else:
        convergence = False
    if verbose:
        print("\tConverged:", convergence, "#skipping edges slope:", round(k, 5),
              "RMS(Pn-Pn+1) slope:", round(k2, 5))
    return convergence

# sample and update model when necessary


def sampling(network, edge2Module, edge2Patients, nOnesPerPatientInModules, moduleSizes,
             moduleOneFreqs, p0, match_score, mismatch_score, bK_1, alpha=0.1, beta_K=1.0,
             max_n_steps=100, n_steps_averaged=20, n_points_fit=10, tol=0.1,
             n_steps_for_convergence=5, verbose=True, edge_ordering="nosort"):

    t0 = time.time()
    if verbose:
        print("Start sampling ...", file=sys.stdout)
    edge_order = list(range(0, edge2Patients.shape[0]))
    if edge_ordering == "shuffle":
        # shuffle edges
        random.shuffle(edge_order)
    edge2Module_history = [copy.copy(edge2Module)]
    is_converged = False
    network_edges = list(network.edges(data=True))
    for step in range(1, max_n_steps):
        not_changed_edges = 0
        t_0 = time.time()
        t_1 = t_0
        i = 1
        for edge_index in edge_order:
            n1, n2, data = network_edges[edge_index]
            # adjust LogP and sample a new module
            p = adjust_lp(data['log_p'], n_exp_orders=7)
            curr_module = data['m']
            edge_ndx = data['e']
            new_module = np.random.choice(data['modules'], p=p)

            # update network and matrices if necessary
            if new_module != curr_module:
                apply_changes(network, n1, n2, edge_ndx, curr_module, new_module,
                              edge2Patients, edge2Module, nOnesPerPatientInModules, moduleSizes,
                              moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
                              alpha=alpha, beta_K=beta_K)

            else:
                not_changed_edges += 1
            i += 1
            if i % 10000 == 1:
                if verbose:
                    print(i, "\t\tedges processed in", round(
                        time.time() - t_1, 1), "s runtime...", file=sys.stdout)
                not_changed_edges = 0
                t_1 = time.time()
        if verbose:
            try:
                tmp = round(1.0*not_changed_edges/len(edge_order), 3)
                print("\tstep ", step, "\t% edges not changed", tmp,
                      "\truntime:", round(time.time() - t_0, 1), "s",
                      file=sys.stdout)
            except ZeroDivisionError:
                print("\tstep ", step, "\t% edges not changed", "\truntime:",
                      round(time.time() - t_0, 1), "s", file=sys.stdout)

        edge2Module_history.append(copy.copy(edge2Module))
        if step == n_steps_averaged:
            is_converged = False
            n_times_cc_fulfilled = 0
            labels = np.asarray(
                edge2Module_history[step-n_steps_averaged:step])
            P_prev = collect_all_p(labels)
            P_diffs = []
            n_skipping_edges = []
            n_skipping_edges.append(len(P_prev.keys()))
        if step > n_steps_averaged:
            labels = np.asarray(
                edge2Module_history[step-n_steps_averaged:step])
            P = collect_all_p(labels)
            P_diff = calc_RMSD(copy.copy(P), copy.copy(P_prev))
            P_diffs.append(P_diff)
            n_skipping_edges.append(len(P.keys()))
            P_prev = P
        if step >= n_steps_averaged + n_points_fit:
            P_diffs_range = min(P_diffs), max(P_diffs)
            n_skipping_edges_range = min(
                n_skipping_edges), max(n_skipping_edges)
            # check convergence condition
            is_converged = check_convergence_conditions(n_skipping_edges[-n_points_fit:],
                                                        n_skipping_edges_range,
                                                        P_diffs[-n_points_fit:],
                                                        P_diffs_range,
                                                        step,
                                                        tol=tol,
                                                        verbose=verbose)
        if is_converged:
            n_times_cc_fulfilled += 1
        else:
            n_times_cc_fulfilled = 0

        if n_times_cc_fulfilled == n_steps_for_convergence:  # stop if convergence is True for the last n steps
            # define how many the last steps to consider
            n_final_steps = n_points_fit+n_steps_for_convergence
            if verbose:
                print("time:\tSampling (%s steps) fininshed in %s s. Model converged." % (
                    len(edge2Module_history), round(time.time()-t0, 2)), file=sys.stdout)
            return edge2Module_history, n_final_steps, n_skipping_edges, P_diffs

    n_final_steps = n_steps_for_convergence
    if verbose:
        print("time:\tSampling (%s steps) fininshed in %s s. Model did not converge." % (
            len(edge2Module_history), round(time.time()-t0, 2)), file=sys.stdout)

    return edge2Module_history, n_final_steps, n_skipping_edges, P_diffs


def apply_changes(network, n1, n2, edge_ndx, curr_module, new_module,
                  edge2Patients, edge2Module, nOnesPerPatientInModules, moduleSizes,
                  moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
                  alpha=1.0, beta_K=1.0, no_LP_update=False):
    '''Moves the edge from current module to the new one
    and updates network, nOnesPerPatientInModules and moduleSizes respectively.'''
    # update the edge module membership

    network[n1][n2]['m'] = new_module
    edge2Module[edge_ndx] = new_module
    # for this edge no probabilities change

    # reduce curr_module size and nOnesPerPatientInModules
    edge_vector = edge2Patients[edge_ndx, ]
    nOnesPerPatientInModules[curr_module,
                             ] = nOnesPerPatientInModules[curr_module, ] - edge_vector
    moduleSizes[curr_module, ] -= 1

    # increase new_module
    nOnesPerPatientInModules[new_module,
                             ] = nOnesPerPatientInModules[new_module, ] + edge_vector
    moduleSizes[new_module, ] += 1

    # update LPs for all edges contacting curr and new modules, except skipping edge
    # for affected edges, calcualte only probabilities regarding curr and new modules
    if not no_LP_update:
        for n in [n1, n2]:
            for n3 in network[n].keys():
                if not n3 == n1 and not n3 == n2:  # for
                    data = network[n][n3]
                    m = data['m']
                    e = data['e']
                    # update LP for new_module
                    lp = calc_lp(e, m, new_module, edge2Patients, nOnesPerPatientInModules, moduleSizes,
                                 moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
                                 alpha=alpha, beta_K=beta_K)
                    # find index of targe module or add it if did not exist
                    if new_module in data['modules']:
                        m_ndx = data['modules'].index(new_module)
                        data['log_p'][m_ndx] = lp

                    else:  # if it is a novel connection to a new_module, append it to the end of a list
                        data['modules'].append(new_module)
                        data['log_p'].append(lp)

                    # update LP for curr_module
                    # check if an edge is still connected with curr_module
                    still_connected = False
                    # iterate all edges adjacent to affected
                    for n1_, n2_, data_ in network.edges([n, n3], data=True):
                        # if there is still one adjacent edge from curr_module or with index e==curr_module , update LP
                        if data_['m'] == curr_module or data_['e'] == curr_module:
                            still_connected = True
                            lp = calc_lp(e, m, curr_module, edge2Patients, nOnesPerPatientInModules, moduleSizes,
                                         moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
                                         alpha=alpha, beta_K=beta_K)
                            m_ndx = data['modules'].index(curr_module)
                            data['log_p'][m_ndx] = lp

                            break

                    # if not connected, remove curr_m from the list
                    if not still_connected:
                        m_ndx = data['modules'].index(curr_module)
                        del data['modules'][m_ndx]
                        del data['log_p'][m_ndx]


def plot_convergence(n_skipping_edges, P_diffs, thr_step, n_steps_averaged, outfile=""):
    # plots numnber of oscilating edges and RMS(Pn-Pn+1)
    steps = range(n_steps_averaged, n_steps_averaged+len(n_skipping_edges))
    fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
    axarr[0].set_title("Model convergence")
    axarr[0].plot(steps, n_skipping_edges, 'b.-')
    axarr[0].axvline(thr_step, color="red", linestyle='--')
    axarr[0].set_ylabel("# edges oscilating on the last " +
                        str(int(n_steps_averaged))+" steps")
    steps = range(n_steps_averaged, n_steps_averaged+len(P_diffs))
    axarr[1].plot(steps, P_diffs, 'b.-')
    axarr[1].set_xlabel('step')
    axarr[1].axvline(thr_step, color="red", linestyle='--')
    tmp = axarr[1].set_ylabel("RMS(Pn-Pn+1)")
    if outfile:
        plt.savefig(outfile, transparent=True)


def get_consensus_modules(edge2module_history, network, edge2Patients, edge2Module,
                          nOnesPerPatientInModules, moduleSizes, moduleOneFreqs, p0, match_score, mismatch_score,
                          bK_1, alpha=1.0, beta_K=1.0, verbose=True):
    consensus_edge2module = []
    labels = np.asarray(edge2module_history)

    # identify modules which edges ocsilate
    edges = list(network.edges())
    for i in range(0, len(edges)):
        unique, counts = np.unique(labels[:, i], return_counts=True)
        if len(unique) > 1:
            counts = np.array(counts)
            new_ndx = unique[np.argmax(counts)]
            if float(max(counts))/labels.shape[0] < 0.5:
                if verbose:
                    print("Warning: less than 50% of time in the most frequent module\n\tedge:", i,
                          "counts:", counts, "\n\tlabels:", ",".join(map(str, unique)), file=sys.stdout)
            consensus_edge2module.append(new_ndx)
        else:
            consensus_edge2module.append(unique[0])

    # construct consensus edge-to-module membership
    i = 0
    changed_edges = 0
    for i in range(0, len(consensus_edge2module)):
        curr_module = edge2Module[i]
        new_module = consensus_edge2module[i]
        if curr_module != new_module:
            changed_edges += 1
            n1, n2 = edges[i]
            apply_changes(network, n1, n2, i, curr_module, new_module,
                          edge2Patients, edge2Module, nOnesPerPatientInModules, moduleSizes,
                          moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
                          alpha=alpha, beta_K=beta_K, no_LP_update=True)
    if verbose:
        print(changed_edges,
              "edges changed their module membership after taking consensus.")
        print("Empty modules:", len([x for x in moduleSizes if x == 0]),
              "\nNon-empty modules:", len([x for x in moduleSizes if x > 0]))
    return consensus_edge2module, network, edge2Patients, nOnesPerPatientInModules, moduleSizes, moduleOneFreqs

################################## 3. Post-processing ####################


def get_genes(mid, edge2module, edges):
    ndx = [i for i, j in enumerate(edge2module) if j == mid]
    genes = []
    for edge in [edges[i] for i in ndx]:
        genes.append(edge[0])
        genes.append(edge[1])
    genes = list(set(genes))
    return genes


def genesets2biclusters(network, exprs_np, exprs_data, moduleSizes, consensus_edge2module,
                        min_SNR=0.5, direction="UP", min_n_samples=10,
                        verbose=True):
    # Identify optimal patient sets for each module: split patients into two sets in a subspace of each module
    # Filter out bad biclusters with too few genes or samples, or with low SNR
    t0 = time.time()
    network_edges = list(network.edges())
    filtered_bics = []
    few_genes = 0
    empty_bics = 0
    wrong_sample_number = 0
    low_SNR = 0

    for mid in range(0, len(moduleSizes)):
        if moduleSizes[mid] > 1:  # exclude biclusters with too few genes
            genes = get_genes(mid, consensus_edge2module, network_edges)
            bic = identify_opt_sample_set(genes, exprs_np, exprs_data,
                                          direction=direction,
                                          min_n_samples=min_n_samples)
            avgSNR = bic["avgSNR"]
            if avgSNR == -1:  # exclude biclusters with too few samples
                wrong_sample_number += 1
            elif avgSNR < min_SNR:  # exclude biclusters with low avg. SNR
                low_SNR += 1
            else:
                bic["id"] = mid
                filtered_bics.append(bic)
        elif moduleSizes[mid] > 0:
            few_genes += 1
        else:
            empty_bics += 1

    if verbose:
        print("time:\tIdentified optimal sample sets for %s modules in %s s." %
              (len(moduleSizes), round(time.time()-t0, 2)))

        print("\tEmpty modules:", few_genes, file=sys.stdout)
        print("\tModules with just 1 edge:", few_genes, file=sys.stdout)
        print("\tModules with not enough or too many samples:",
              wrong_sample_number, file=sys.stdout)
        print("\tModules not passed avg. |SNR| threshold:",
              low_SNR, file=sys.stdout)

        print("Passed modules with >= 2 edges and >= %s samples: %s" %
              (min_n_samples, len(filtered_bics)), file=sys.stdout)
    return filtered_bics


def calc_J(bic, bic2, all_samples):
    s1 = bic["samples"]
    s2 = bic2["samples"]
    if bic["direction"] != bic2["direction"]:
        s2 = all_samples.difference(s2)
    sh_samples = s1.intersection(s2)
    u_samples = s1.union(s2)
    J = 1.0*len(sh_samples)/len(u_samples)
    return J


def calc_overlap_pval_J(bic, bic2, all_samples):
    s1 = bic["samples"]
    s2 = bic2["samples"]
    if bic["direction"] != bic2["direction"]:
        s2 = all_samples.difference(s2)
    s1_s2 = len(s1.intersection(s2))
    s1_only = len(s1.difference(s2))
    s2_only = len(s2.difference(s1))
    p_val = pvalue(s1_s2, s1_only, s2_only, len(
        all_samples)-s1_s2-s1_only-s2_only).right_tail
    J = 1.0*s1_s2/(s1_s2+s1_only+s2_only)
    return p_val, J


def merge_biclusters(biclusters, exprs, exprs_data, min_n_samples=5,
                     verbose=True, min_SNR=0.5, J_threshold=0.25, pval_threshold=0.05,  report_merging=False):
    t0 = time.time()

    n_merging_cycles = 0
    n_merges = 0
    n_bics = len(biclusters)
    bics = dict(zip(range(0, len(biclusters)), biclusters))

    all_samples = set(range(0, exprs.shape[1]))
    candidates = {}
    for i in bics.keys():
        bic = bics[i]
        for j in bics.keys():
            if j > i:
                bic2 = bics[j]
                p_val, J = calc_overlap_pval_J(bic, bic2, all_samples)
                p_val = p_val*n_bics
                if p_val < pval_threshold:  # and J>J_threshold:
                    candidates[(i, j)] = p_val, J

    nothing_to_merge = False
    while not nothing_to_merge:
        n_merging_cycles += 1
        # take min p-value pair
        opt_pval, opt_J = pval_threshold, 0.0
        opt_pair = -1, -1

        for pair in candidates.keys():
            p_val, J = candidates[pair]
            if p_val < opt_pval:
                opt_pval, opt_J = p_val, J
                opt_pair = pair
            elif p_val == opt_pval:  # take max(J) in case of ties
                if J > opt_J:
                    opt_pval, opt_J = p_val, J
                    opt_pair = pair

        if opt_pair[0] == -1:
            nothing_to_merge = True
        else:
            opt_i, opt_j = opt_pair
            if report_merging and verbose:
                print("\t\ttry merging %s:%sx%s  (%s,%s) + %s:%sx%s  (%s,%s)" % (bics[opt_i]["id"], bics[opt_i]["n_genes"],
                                                                                 bics[opt_i]["n_samples"],
                                                                                 round(
                                                                                     bics[opt_i]["avgSNR"], 2),
                                                                                 bics[opt_i]["direction"],
                                                                                 bics[opt_j]["id"], bics[opt_j]["n_genes"],
                                                                                 bics[opt_j]["n_samples"],
                                                                                 round(
                                                                                     bics[opt_j]["avgSNR"], 2),
                                                                                 bics[opt_j]["direction"]))

            # try creating a new bicsluter from bic and bic2
            genes = list(bics[opt_i]["genes"] | bics[opt_j]["genes"])
            new_bic = identify_opt_sample_set(genes, exprs, exprs_data,
                                              direction=bics[opt_i]["direction"],
                                              min_n_samples=min_n_samples)
            if new_bic["avgSNR"] == -1 and bics[opt_i]["direction"] != bics[opt_j]["direction"]:
                new_bic = identify_opt_sample_set(
                    genes, exprs, exprs_data, direction=bics[opt_j]["direction"], min_n_samples=min_n_samples)

            avgSNR = new_bic["avgSNR"]
            if avgSNR >= min_SNR:
                # place new_bic to ith bic
                new_bic["id"] = bics[opt_i]["id"]
                if verbose:
                    substitution = (bics[opt_i]["id"], len(bics[opt_i]["genes"]), len(bics[opt_i]["samples"]),
                                    round(bics[opt_i]["avgSNR"],
                                          2), bics[opt_i]["direction"],
                                    bics[opt_j]["id"], len(
                        bics[opt_j]["genes"]),
                        len(bics[opt_j]["samples"]),
                        round(bics[opt_j]["avgSNR"],
                              2), bics[opt_j]["direction"],
                        round(new_bic["avgSNR"], 2), len(
                        new_bic["genes"]),
                        len(new_bic["samples"]))
                    print(
                        "\tMerge biclusters %s:%sx%s (%s,%s) and %s:%sx%s  (%s,%s) --> %s SNR and %sx%s" % substitution)
                new_bic["n_genes"] = len(new_bic["genes"])
                new_bic["n_samples"] = len(new_bic["samples"])
                n_merges += 1  # number of successfull merges

                bics[opt_i] = new_bic
                # deleted data for ith and jth biclusters
                candidates_keys = list(candidates.keys())
                for k in range(len(candidates_keys)):
                    (i, j) = candidates_keys[k]
                    if i == opt_j or j == opt_j:
                        del candidates[(i, j)]
                # remove j-th bics jth column and index
                del bics[opt_j]
                n_bics = len(bics)
                for j in bics.keys():
                    if j != opt_i:
                        J = calc_J(new_bic, bics[j], all_samples)
                        p_val, J = calc_overlap_pval_J(
                            new_bic, bics[j], all_samples)
                        p_val = p_val*n_bics
                        if p_val < pval_threshold:  # and J>J_threshold:
                            if opt_i < j:
                                candidates[(opt_i, j)] = p_val, J
                            else:
                                candidates[(j, opt_i)] = p_val, J
            else:
                # set J for this pair to 0
                if report_merging and verbose:
                    print("\t\tSNR=", round(avgSNR, 2), "<",
                          round(min_SNR, 2), "--> no merge")
                candidates[(opt_i, opt_j)] = pval_threshold, 0.0

    merged_bics = bics.values()
    if verbose:
        print("time:\tMerging (%s cycles, %s successful merges) fininshed in %s s." % (
            n_merging_cycles, n_merges, round(time.time()-t0, 2)), file=sys.stdout)
        print("Modules remaining after merging:",
              len(merged_bics), file=sys.stdout)
    return merged_bics


###### save and read modules #####
def write_bic_table(resulting_bics, results_file_name):
    resulting_bics_df = pd.DataFrame.from_dict(resulting_bics)
    if len(resulting_bics) == 0:
        pass
    else:
        resulting_bics_df["genes"] = resulting_bics_df["genes"].apply(
            lambda x: " ".join(map(str, x)))
        resulting_bics_df["samples"] = resulting_bics_df["samples"].apply(
            lambda x: " ".join(map(str, x)))
        resulting_bics_df = resulting_bics_df[[
            "id", "avgSNR", "n_genes", "n_samples", "direction", "genes", "samples"]]
        resulting_bics_df.sort_values(
            by=["avgSNR", "n_genes", "n_samples"], inplace=True, ascending=False)
        resulting_bics_df["id"] = range(0, resulting_bics_df.shape[0])
    resulting_bics_df.to_csv(results_file_name, sep="\t", index=False)


def read_bic_table(results_file_name):
    if not os.path.exists(results_file_name):
        return pd.DataFrame()
    resulting_bics = pd.read_csv(results_file_name, sep="\t")
    if len(resulting_bics) == 0:
        return pd.DataFrame()
    else:
        resulting_bics["genes"] = resulting_bics["genes"].apply(
            lambda x: set(x.split(" ")))
        resulting_bics["samples"] = resulting_bics["samples"].apply(
            lambda x: set(x.split(" ")))
    # resulting_bics.set_index("id",inplace=True)

    return resulting_bics


def run_DESMOND(exprs_file, network_file,
                direction="UP", min_n_samples=-1,
                p_val=0.01, alpha=1.0, beta_K=1.0, q=0.5,
                out_dir="./", basename=False,
                max_n_steps=200, n_steps_averaged=20, n_steps_for_convergence=5,
                force=False, plot_all=True, report_merging=False, verbose=True):

    import datetime
    from partial_RRHO import relabel_exprs_and_network, define_SNR_threshold

    start_time = time.time()

    if basename:
        basename = basename
    else:
        [date_h, mins] = str(datetime.datetime.today()).split(":")[:2]
        [date, hs] = date_h.split()
        basename = "results_"+hs+":"+mins+"_"+date

    suffix = ".alpha="+str(alpha)+",beta_K="+str(beta_K) + \
        ",direction="+direction+",p_val="+str(p_val)+",q="+str(q)
    if verbose:
        print("Will save output files to:", out_dir +
              basename + suffix, file=sys.stdout)

    ### 1. Assigning samples to edges ###
    # read inputs
    exprs, network = prepare_input_data(
        exprs_file, network_file, verbose=verbose, min_n_nodes=3)

    # define minimal number of patients in a module
    if min_n_samples == -1:
        # set to max(10, 5% of the cohort)
        min_n_samples = int(max(10, 0.05*exprs.shape[1]))
    if verbose:
        print("Mininal number of samples in a module:",
              min_n_samples, file=sys.stdout)

    # change gene and sample names to ints
    exprs, network, ints2g_names, ints2s_names = relabel_exprs_and_network(
        exprs, network)
    exprs_np = exprs.values
    exprs_sums = exprs_np.sum(axis=1)
    exprs_sq_sums = np.square(exprs_np).sum(axis=1)
    N = exprs.shape[1]
    exprs_data = N, exprs_sums, exprs_sq_sums

    # read min_SNR from file or determine it from avg.SNR distributions among 1000 random edges
    snr_file = out_dir+basename + ",q="+str(q) + ".SNR_threshold.txt"
    if not force and os.path.exists(snr_file) and os.path.getsize(snr_file) > 0:
        f = open(snr_file, "r")
        min_SNR = f.readlines()[0]
        f.close()
        try:
            min_SNR = float(min_SNR)
            if verbose:
                print("Using pre-computed SNR threshold:\t%s (q=%s)" %
                      (min_SNR, q), file=sys.stdout)
        except:
            min_SNR = define_SNR_threshold(exprs_np, exprs_data, network, q, snr_file,
                                           min_n_samples=min_n_samples, verbose=verbose)
    else:
        min_SNR = define_SNR_threshold(exprs_np, exprs_data, network, q, snr_file,
                                       min_n_samples=min_n_samples, verbose=verbose)

    # first check if the network already exists and try loading it
    network_with_samples_file = out_dir+basename + ".direction="+direction + \
        ",p_val="+str(p_val)+",q="+str(q)+",min_ns=" + \
        str(min_n_samples)+".network.txt"

    if not force and os.path.exists(network_with_samples_file):
        from method import print_network_stats, load_network
        network = load_network(network_with_samples_file, verbose=verbose)
        print_network_stats(network)
    else:
        from partial_RRHO import precompute_RRHO_thresholds,  expression_profiles2nodes, assign_patients2edges
        from method import save_network
        # assign expression vectors on nodes
        network = expression_profiles2nodes(network, exprs, direction)

        # define step for RRHO
        fixed_step = int(max(1, 0.01*exprs.shape[1]))  # 5-10-20 ~15
        if verbose:
            print("Fixed step for RRHO selected:", fixed_step, file=sys.stdout)

        rrho_thresholds = precompute_RRHO_thresholds(
            exprs, fixed_step=fixed_step, significance_thr=p_val)

        #  assign patients on edges
        network = assign_patients2edges(network, rrho_thresholds, min_SNR=min_SNR, min_n_samples=min_n_samples,
                                        fixed_step=fixed_step, verbose=verbose)

        # save the network with patients on edges
        save_network(network, network_with_samples_file, verbose=verbose)

    if plot_all:
        from partial_RRHO import plot_edge2sample_dist
        plot_outfile = out_dir + basename + suffix+".n_samples_on_edges.svg"
        plot_edge2sample_dist(network, plot_outfile)

    ### 2. Edge clustering ###

    # simplifying probability calculations
    max_log_float = np.log(np.finfo(np.float64).max)
    n_exp_orders = 7  # ~1000 times
    p0 = N*np.log(0.5)+np.log(beta_K)
    match_score = np.log((alpha*0.5+1)/(alpha))
    mismatch_score = np.log((alpha*0.5+0)/alpha)
    bK_1 = math.log(1+beta_K)

    # set initial model state
    [moduleSizes, edge2Patients, nOnesPerPatientInModules, edge2Module, moduleOneFreqs, network] = set_initial_conditions(network, p0, match_score, mismatch_score, bK_1, N,
                                                                                                                          alpha=alpha, beta_K=beta_K, verbose=verbose)

    # sampling
    edge2Module_history, n_final_steps, n_skipping_edges, P_diffs = sampling(network, edge2Module, edge2Patients,
                                                                             nOnesPerPatientInModules, moduleSizes, moduleOneFreqs,
                                                                             p0, match_score, mismatch_score, bK_1,
                                                                             alpha=alpha, beta_K=beta_K,
                                                                             max_n_steps=max_n_steps, n_steps_averaged=n_steps_averaged,
                                                                             n_points_fit=10, tol=0.1,
                                                                             n_steps_for_convergence=n_steps_for_convergence,
                                                                             edge_ordering="shuffle", verbose=verbose)
    if plot_all:
        from method import plot_convergence
        plot_outfile = out_dir + basename + suffix+",ns_max=" + str(max_n_steps) + ",ns_avg=" + str(
            n_steps_averaged) + ",ns_c="+str(n_steps_for_convergence) + ".convergence.svg"
        plot_convergence(n_skipping_edges, P_diffs, len(edge2Module_history)-n_final_steps,
                         n_steps_averaged, outfile=plot_outfile)

    # take the last (n_points_fit+n_steps_for_convergence) steps
    edge2Module_history = edge2Module_history[-n_final_steps:]

    # get consensus edge-to-module membership
    consensus_edge2module, network, edge2Patients, nOnesPerPatientInModules, moduleSizes, moduleOneFreqs = get_consensus_modules(edge2Module_history, network,
                                                                                                                                 edge2Patients, edge2Module, nOnesPerPatientInModules, moduleSizes,
                                                                                                                                 moduleOneFreqs, p0, match_score, mismatch_score, bK_1,
                                                                                                                                 alpha=alpha, beta_K=beta_K, verbose=verbose)
    #### 3. Define biclusters and merge modules  ####

    # identify optimal patient sets for each module: split patients into two sets in a subspace of each module
    # filter out bad biclusters with too few genes or samples, or with low SNR
    filtered_bics = genesets2biclusters(network, exprs_np, exprs_data, moduleSizes, consensus_edge2module,
                                        min_SNR=min_SNR, direction=direction, min_n_samples=min_n_samples,
                                        verbose=verbose)

    # merge remaining biclusters
    merged_bics = merge_biclusters(filtered_bics, exprs_np, exprs_data,
                                   min_n_samples=min_n_samples, min_SNR=min_SNR,
                                   verbose=verbose, report_merging=report_merging)

    # print info on merged biclusters
    if verbose:
        i = 0
        print()
        for bic in merged_bics:
            bic["id"] = i
            i += 1
            bic["genes"] = {ints2g_names[x] for x in bic["genes"]}
            bic["samples"] = {ints2s_names[x] for x in bic["samples"]}
            print("\t".join(map(str, [bic["id"], bic["n_genes"], bic["n_samples"],
                                      bic["avgSNR"], bic["direction"]])), file=sys.stdout)

    # save results
    result_file_name = out_dir+basename+suffix
    write_bic_table(merged_bics,
                    result_file_name+".biclusters.tsv")
    if verbose:
        print("Total runtime:", round(time.time()-start_time, 2), file=sys.stdout)
    return merged_bics
