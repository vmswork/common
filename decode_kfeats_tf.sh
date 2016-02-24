#! /bin/bash

# Decoding with Tensorflow neural network directly from Kaldi features

. cmd.sh
. path.sh

data=data/tr05_multi_beamformit_5mics_fbank_deltas_nocmvn_2utt  #$1
nnet=exp/dnn_pretrain_tr05_multi_beamformit_5mics_fbank_nocmvn_broken/tr_splice5_cmvn-g.nnet #$2

model=~/kaldi/egs/chime3/s5_v/exp/tri3b_tr05_multi_beamformit_5mics/final.mdl
fst=~/kaldi/egs/chime3/s5_v/exp/tri3b_tr05_multi_beamformit_5mics/graph_tgpr_5k/HCLG.fst
out=ark,t:-
symtab=~/kaldi/egs/chime3/s5_v/exp/tri3b_tr05_multi_beamformit_5mics/graph_tgpr_5k/words.txt

loglikes=ark,t:-

# Apply global cmvn and splicing
./local/common/apply_splice_cmvn_g.sh $data $nnet ark,t:- | \
   python local/common/tf_inference.py --feats_rspec 'ark,t:-' --prob_wspec 'ark,t:-' | \
    decode-faster-mapped $model $fst $loglikes $out | \
      utils/int2sym.pl -f 2- $symtab
    

exit
# Calculate posteriors and convert them to likelihoods and decode
python local/common/tf_inference.py --prob_wspec 'ark,t:-'
# Score


exit

loglikes=scp:2.scp
loglikes=ark,t:-
model=~/kaldi/egs/chime3/s5/exp/blstm_fmllr-ivec-cmvn_10/final.mdl
fst=~/kaldi/egs/chime3/s5/exp_lm/best_fair_dnn/graph_4_GT_default_pruned_c_pr/HCLG.fst
out=ark,t:-
symtab=~/kaldi/egs/chime3/s5/exp_lm/best_fair_dnn/graph_4_GT_default_pruned_c_pr/words.txt

rm 2.ints

cat 2.ark | decode-faster-mapped $model $fst $loglikes $out > 2.ints
cat 2.ints  | ../../../utils/int2sym.pl -f 2- $symtab