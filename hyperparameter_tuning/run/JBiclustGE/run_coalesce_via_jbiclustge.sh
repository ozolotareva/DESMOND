# usage: ./run_coalesce_via_jbiclustge.sh exprs_file basename prob_gene pvalue_cond pvalue_correl zscore_cond

#/opt/jbiclustge-cli/jbiclustge-cli.sh -newprofile

root_dir=`pwd`;

exprs_file=$1;
basename=$2;
prob_gene=$3
pvalue_cond=$4;
pvalue_correl=$5;
zscore_cond=$6;

echo $exprs_file $basename $prob_gene $pvalue_cond $pvalue_correl $zscore_cond;
file_folder=$basename;
param_folder="prob_gene="$prob_gene",pvalue_cond="$pvalue_cond",pvalue_correl="$pvalue_correl",zscore_cond="$zscore_cond"/";

mkdir -p $param_folder;
mkdir $file_folder;
mkdir $file_folder/algorithms/;
# modify algorithm template and save
cat algorithm_templates/coalesce_configuration.conf  | sed -e 's/@prob_gene@/'$prob_gene'/' \
-e 's/@pvalue_cond@/'$pvalue_cond'/' -e 's/@pvalue_correl@/'$pvalue_correl'/' \
-e 's/@zscore_cond@/'$zscore_cond'/' > $file_folder/algorithms/coalesce_configuration.conf;
# softlink to exprs dataset file
ln -s ../$exprs_file $file_folder/dataset.tsv;
ln -s ../profile.conf $file_folder/profile.conf;
# run JbiclustGE
/opt/jbiclustge-cli/jbiclustge-cli.sh -run $root_dir/$file_folder > $file_folder/LOG 2> $file_folder/ERR;
# move the results to the $param_folder
mv $file_folder $param_folder;
