gpu=(6)
items=(5)
length=${#items[@]}
for ((i = 0;i<length;i++));do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    nohup python ../src/evaluation/eval.py  \
        --model_name_or_path  \
        --input_file "Your evaluation input jsonl format file" \
        --output_path "Your evaluation output jsonl format file"\
        --retrieval_augment \
        --max_new_tokens 100  \
        --metric f1  \
        --task wow  \
        --top_n ${items[$i]}  \
        --vllm \
        --case_num -1  \
        --llama_style \
        --user_chat_template  > wow_"${items[$i]}"_psg.out  2>&1 &
done


