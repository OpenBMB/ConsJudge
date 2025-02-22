INPUT_JSONL_PATH="../data/ConsJudge_train/test.choices.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/test.choices_eval.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/hybrid_evaluation.py \
    --input_jsonlpath "$INPUT_JSONL_PATH" \
    --output_jsonlpath "$OUTPUT_JSONL_PATH" \
    --model_path "$MODEl_PATH"


INPUT_JSONL_PATH="../data/ConsJudge_train/test.choices_eval.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/test.choices_final.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/embedding_similarity.py \
    --input_jsonlpath "$INPUT_JSONL_PATH" \
    --output_jsonlpath "$OUTPUT_JSONL_PATH" \
    --model_path "$MODEl_PATH"