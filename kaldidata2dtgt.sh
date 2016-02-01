#! /bin/bash

. cmd.sh
. path.sh

function kd2dtgt() {

    mdl=exp/$1/final.mdl
    data=$2
    feats="scp:"$data/feats.scp
    alidir=exp/$1
    ali="ark:gunzip -c $alidir/ali.*.gz |"
    dat=$data".dat"
    tgt=$data".tgt"

    local/nnet-stc-convert-features $mdl $feats "$ali" $dat $tgt
}

kd2dtgt  tri3c1_sil1ns_all0711_sil100BN80_all3007fix_ali data/train_10k_tr90




