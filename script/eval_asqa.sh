gpu=(4)
items=(5)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    nohup python ../src/evaluation/eval.py  \
        --model_name_or_path  \
        --input_file  \
        --output  \
        --retrieval_augment \
        --use_lora \
        --max_new_tokens 100  \
        --metric rouge  \
        --task asqa \
        --top_n ${items[$i]}  \
        --vllm \
        --rerank \
        --user_chat_template  > asqa_"${items[$i]}"psg.out  2>&1 &
done