# usage: ./run_debi_via_jbiclustge.sh exprs_file basename binarization_level[1.0] pattern_of_regulation[u|d]

#/opt/jbiclustge-cli/jbiclustge-cli.sh -newprofile

root_dir=`pwd`;

exprs_file=$1;
basename=$2;
b=$3;
p=$4;

echo $exprs_file $basename $b $p
file_folder=$basename;
param_folder="p="$p",b="$b;

mkdir -p $param_folder;
mkdir $file_folder;
mkdir $file_folder/algorithms/;
# modify algorithm template and save
cat algorithm_templates/debi_configuration.conf  | sed -e 's/@binarization_level@/'$b'/'\
 | sed -e 's/@pattern_of_regulation@/'$p'/'  > $file_folder/algorithms/debi_configuration.conf;
# softlink to exprs dataset file
ln -s ../$exprs_file $file_folder/dataset.tsv;
ln -s ../profile.conf $file_folder/profile.conf;
# run JbiclustGE
/opt/jbiclustge-cli/jbiclustge-cli.sh -run $root_dir/$file_folder > $file_folder/LOG 2> $file_folder/ERR;
# move the results to the $param_folder
mv $file_folder $param_folder;
