#! /bin/bash

ali=$1
nnet=$2
data=$3
setname=$4

[ -z $setname ] && setname=train

# sudo unlink /usr/local/cuda
# sudo ln -s /usr/local/cuda-7.5 /usr/local/cuda
local/common/apply_splice_cmvn_g.sh $data $nnet
# sudo unlink /usr/local/cuda
# sudo ln -s /usr/local/cuda-7.0 /usr/local/cuda

local/common/kaldidata2dtgt.sh $ali $data "_splice_cmvng" 
python local/common/dtgt2recs.py --directory ${data} --features ${data}.dat --labels ${data}.tgt --setname $setname

