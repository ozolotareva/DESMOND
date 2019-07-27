# DESMOND

DESMOND is a method for identification of **D**ifferentially **E**xpre**S**sed gene **MO**dules i**N** **D**iseases. 

DESMOND accepts gene expression matrix and gene interaction network and identifies connected sets of genes up- or down-regulated in subsets of samples.

![alt text](https://github.com/ozolotareva/DESMOND/blob/master/poster/DESMOND_abstract.png)


### Input

 * a matrix of normalized gene expressions; first row and column contain gene and sample names respectively
 * a network of gene interactions; list of edges in a format gene1\tgene2 or network in NDEx format https://home.ndexbio.org/about-ndex/
 
### Usage example

```
python DESMOND.py --exprs $exprs --network $network  --basename $proj_name --out_dir $outdir \
--alpha 0.5 --beta_K 1.0 --p_val 0.005 --min_SNR 0.5 \
--verbose --plot_all

```
### Output
 * \*.biclusters.txt - list of identified biclusters
 * \*.convergence.svg - model convergence plot 
 * \*.initial_state.pickle, \*.network.txt, \*.e2m_history.pickle - temporary files
