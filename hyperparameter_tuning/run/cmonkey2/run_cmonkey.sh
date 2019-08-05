n_steps=300
for g in 5 10 20 50 100; do
 for p in 10 20 50 100; do
  basename=$g','$p;
  network='../../simulated_datasets/networks/simulated.N=10.Mu=2.0.GxP='$basename'.overlap=TT.tab';
  exprs='../../simulated_datasets/exprs/simulated.N=10.Mu=2.0.GxP='$basename'.overlap=TT.exprs.tsv';
  for iter in `seq 1 10`; do
   mkdir -p $basename'_BSCM_'$iter;
   ../bin/cmonkey2.sh --organism hsa --string $network --nomotifs --nooperons --nonetworks --logfile $basename.LOG --num_iterations $n_steps --use_BSCM --interactive  $exprs  2>> $basename.ERR;
   ../bin/cmonkey2.sh --organism hsa --string $network --nomotifs --nooperons --nonetworks --logfile $basename.LOG --num_iterations $n_steps --use_BSCM $exprs  2>> $basename.ERR;
   mv $basename.LOG $basename.ERR cache out $basename'_BSCM_'$iter;
   mkdir -p $basename'_noBSCM_'$iter;
   ../bin/cmonkey2.sh --organism hsa --string $network --nomotifs --nooperons --nonetworks --logfile $basename.LOG --num_iterations $n_steps --interactive $exprs  2>> $basename.ERR;
   ../bin/cmonkey2.sh --organism hsa --string $network --nomotifs --nooperons --nonetworks --logfile $basename.LOG --num_iterations $n_steps $exprs  2>> $basename.ERR;   mv $basename.LOG $basename.ERR cache out $basename'_noBSCM_'$iter;
  done;
 done;
done&
