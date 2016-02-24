#! /bin/bash

. path.sh

function apply_splice_cmvn_g() {
  
  print $1
  print $2
  nnet-forward --use_gpu=yes $1 scp:${2}/feats.scp $3
}
dir=$1 # exp/dnn5b_10k
nnet=$2 # $dir/tr_splice5_cmvn-g.nnet 
out=$3

if [ -z "$out" ]; then
  out="ark,scp:${dir}/feats_splice_cmvng.ark,${dir}/feats_splice_cmvng.scp"
fi

apply_splice_cmvn_g $nnet ${dir} $out