gpu=(1)
items=(5)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    nohup python ../src/evaluation/eval.py  \
        --model_name_or_path  \
        --input_file  \
        --retrieval_augment \
        --max_new_tokens 32  \
        --metric accuracy  \
        --task hotpotqa  \
        --top_n ${items[$i]}  \
        --vllm \
        --user_chat_template  > hotpotqa_"${items[$i]}"_psg.out  2>&1 &
done