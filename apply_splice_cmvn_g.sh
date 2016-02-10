#! /bin/bash

function apply_splice_cmvn_g() {
  
  print $1
  print $2
  nnet-forward --use_gpu=yes $1 scp:${2}/feats.scp ark,scp:${2}/feats_splice_cmvng.ark,${2}/feats_splice_cmvng.scp
}
dir=$1 # exp/dnn5b_10k
nnet=$2 # $dir/tr_splice5_cmvn-g.nnet 
apply_splice_cmvn_g $nnet ${dir}