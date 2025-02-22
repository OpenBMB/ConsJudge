
TEMPERATURE=0.5
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/cpm_2b_0.5.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/minicpm_2b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.6
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/cpm_2b_0.6.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/minicpm_2b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.7
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/cpm_2b_0.7.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/minicpm_2b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.5
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/cpm_4b_0.5.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/minicpm_4b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.6
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/cpm_4b_0.6.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/minicpm_4b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.7
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/cpm_4b_0.7.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/minicpm_4b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.5
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/llama3_8b_0.5.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/llama3_8b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.6
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/llama3_8b_0.6.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/llama3_8b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.7
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/llama3_8b_0.7.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/llama3_8b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.5
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/qwen_14b_0.5.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/qwen_14b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.6
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/qwen_14b_0.6.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/qwen_14b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"

TEMPERATURE=0.7
INPUT_JSONL_PATH="../data/ConsJudge_train/test.jsonl"
OUTPUT_JSONL_PATH="../data/ConsJudge_train/qwen_14b_0.7.jsonl"
MODEl_PATH=""

python ../src/ConsJudge_train/qwen_14b_infer.py \
   --temperature "$TEMPERATURE" \
   --input_jsonlpath "$INPUT_JSONL_PATH" \
   --output_jsonlpath "$OUTPUT_JSONL_PATH" \
   --model_path "$MODEl_PATH"