export CUDA_VISIBLE_DEVICES=3

python sentence_alignment/train_model_script.py --max_evi 8 --train_epoch 5

MODEL_DIR="sentence_alignment/results-model/checkpoint-7500/"
TEST_DATA="dataset/test.json"
OUTPUT_FILE="test_alignment.json"
MAX_EVIDENCE_COUNT=4

python sentence_alignment/predict_model_script.py \
    --model_dir "$MODEL_DIR" \
    --test_data "$TEST_DATA" \
    --output_file "$OUTPUT_FILE" \
    --max_evidence_count "$MAX_EVIDENCE_COUNT"