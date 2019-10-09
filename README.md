# DESMOND

DESMOND is a method for identification of **D**ifferentially **E**xpre**S**sed gene **MO**dules i**N** **D**iseases. 

DESMOND accepts gene expression matrix and gene interaction network and identifies connected sets of genes up- or down-regulated in subsets of samples.

![alt text](https://github.com/ozolotareva/DESMOND/blob/master/poster/DESMOND_abstract.png)




### Input

 * matrix of normalized gene expressions; first row and column contain gene and sample names respectively
 * network of gene interactions; list of edges in a format gene1\tgene2 or network in NDEx format https://home.ndexbio.org/about-ndex/
 
### Usage example

| Disclaimer:  the method is still under development and is not properly tested |
|---|

```
python DESMOND.py --exprs $exprs --network $network  --basename $proj_name --out_dir $outdir \
--alpha 0.5 --p_val 0.01 --q 0.5  --direction [UP|DOWN] --verbose --save-gc >LOG 2>ERR;

```
### Output
 * \*.biclusters.tsv - list of identified biclusters.
 * \*.network.txt  - temporary network file, contains the network with samples assigned on edges. This file is used for restarts with the same network and parameters 'direction', 'p_val', 'min_SNR' and 'min_n_samples'.
 * \*.gene_clusters.txt - gene clusters resulting in sampling phase.
