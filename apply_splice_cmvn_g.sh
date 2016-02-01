#! /bin/bash

. cmd.sh
. path.sh

function apply_splice_cmvn_g() {
  
  nnet-forward --use_gpu=yes $1 scp:${2}/feats.scp ark,scp:${2}/feats_splice_cmvng.ark,${2}/feats_splice_cmvng.scp
}
dir=exp/dnn5b_10k
nnet=$dir/tr_splice5_cmvn-g.nnet 
apply_splice_cmvn_g $nnet data/train_10k