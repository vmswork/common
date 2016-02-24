#! /bin/bash

. ../../../cmd.sh
. ../../../path.sh

loglikes=scp:2.scp
loglikes=ark,t:-
model=~/kaldi/egs/chime3/s5_v/exp/tri3b_tr05_multi_beamformit_5mics/final.mdl
fst=~/kaldi/egs/chime3/s5_v/exp/tri3b_tr05_multi_beamformit_5mics/graph_tgpr_5k/HCLG.fst

out=ark,t:-
symtab=~/kaldi/egs/chime3/s5_v/exp/tri3b_tr05_multi_beamformit_5mics/graph_tgpr_5k/words.txt

fout=t_like_u1
fout=w

rm ${fout}.ints

cat ${fout}.ark | decode-faster-mapped $model $fst $loglikes $out > ${fout}.ints
cat ${fout}.ints  | ../../../utils/int2sym.pl -f 2- $symtab


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