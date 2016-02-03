#! /bin/bash

ali=$1
nnet=$2
data=$3

local/common/apply_splice_cmvn_g.sh $data $nnet
local/common/kaldidata2dtgt.sh $ali $data "_splice_cmvng" 

