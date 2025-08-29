export CUDA_VISIBLE_DEVICES="3"

DATAFILE="test_alignment.json" # the output of evidence alignment
LITERAL="method/result/20250412_123111/log.jsonl" # the output of claim verification
INTENT_MODEL="ft:gpt-4o-mini-2024-07-18:nus-ctic:intent-reproduce:BLToK5NR" # the output model id of intent generator

python -m method.reassessment --datafile "$DATAFILE" --literal "$LITERAL" --intent_model "$INTENT_MODEL"