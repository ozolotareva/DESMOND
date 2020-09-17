# DESMOND

DESMOND is a method for identification of **D**ifferentially **E**xpre**S**sed gene **MO**dules i**N** **D**iseases. 

DESMOND accepts gene expression matrix and gene interaction network and identifies connected sets of genes up- or down-regulated in subsets of samples.

![alt text](https://github.com/ozolotareva/DESMOND/blob/master/poster/DESMOND_abstract.png)



### Input

 * matrix of normalized gene expressions; first row and column contain gene and sample names respectively
 * network of gene interactions; list of edges in a format gene1\tgene2 or network in NDEx format https://home.ndexbio.org/about-ndex/
 
### Usage example

```
python DESMOND.py --exprs $exprs --network $network  --basename $proj_name --out_dir $outdir \
 --alpha $a --p_val $p_val -q $q  --direction UP --verbose  > $outdir/$proj_name.UP.LOG 2> $outdir/$proj_name.UP.ERR;
python DESMOND.py --exprs $exprs --network $network  --basename $proj_name --out_dir $outdir \
 --alpha $a --p_val $p_val -q $q  --direction DOWN --verbose > $outdir/$proj_name.DOWN.LOG 2> $outdir/$proj_name.DOWN.ERR;

# merge up- and down-regulated biclusters and calculate empirical p-values

python post-processing.py --up $outdir/$proj_name.'alpha='$a',beta_K='$b',direction=UP,p_val='$p_val',q='$q.biclusters.tsv \
--down $outdir/$proj_name.'alpha='$a',beta_K='$b',direction=DOWN,p_val='$p_val',q='$q.biclusters.tsv \
--exprs $exprs --network $network -s $outdir/$proj_name',q='$q'.SNR_threshold.txt' \
--out $outdir/$proj_name.'alpha='$a',beta_K='$b',p_val='$p_val',q='$q.biclusters.permutations.tsv  --verbose  > $outdir/$proj_name.permutations.LOG 2> $outdir/$proj_name.permutations.ERR;

```
### Output
 * \*.biclusters.tsv - list of identified biclusters.
 * \*.network.txt  - temporary network file, contains the network with samples assigned on edges. This file is used for restarts with the same network and parameters 'direction', 'p_val', 'min_SNR' and 'min_n_samples'.

### Cite
https://www.biorxiv.org/content/10.1101/2020.04.23.055004v1.full
