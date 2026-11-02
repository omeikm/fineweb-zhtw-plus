OUT="data/filterflag"
DUMP="fineweb2_cmn"

python fineweb2/analyze_flags.py \
  --flags ${OUT}/${DUMP}/5_zhtwplus/rule_flags_${DUMP}.csv \
  --hits  ${OUT}/${DUMP}/5_zhtwplus/keyword_hits_${DUMP}.csv \
  --outdir analysis --topk 30 --n_samples 30