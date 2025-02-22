gpu=(5)
items=(5)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    nohup python ../src/evaluation/eval.py  \
        --model_name_or_path  \
        --input_file  \
        --output_path  \
        --retrieval_augment \
        --max_new_tokens 100  \
        --metric rouge  \
        --task marco  \
        --top_n ${items[$i]}  \
        --case_num -1  \
        --vllm \
        --user_chat_template  > marco_"${items[$i]}"_psg.out  2>&1 &
done