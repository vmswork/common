#! /bin/bash

. cmd.sh
. path.sh

ali=$1
data=$2
suff=$3

function kd2dtgt() {

    mdl=$1/final.mdl
    data=$2
	featsfile=$data/feats"${3}".scp
	if [ ! -f $featsfile ]; then
		echo $featsfile " doesn't exist"
		exit 1
	fi
	[ ! -f $mdl ] && echo $mdl " doesn't exist" && exit 1;
    feats="scp:"$featsfile
    ali="ark:gunzip -c $ali/ali.*.gz |"
    dat=$data".dat"
    tgt=$data".tgt"

    local/common/nnet-convert-features $mdl $feats "$ali" $dat $tgt
}

kd2dtgt $ali $data "$suff"

# kd2dtgt  tri3c1_sil1ns_all0711_sil100BN80_all3007fix_ali data/train_10k_tr90




