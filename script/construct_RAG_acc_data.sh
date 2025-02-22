gpu=()
items=()
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    nohup python  ../src/RAG_train/infer_acc.py \
        --input_data_path  ../data/RAG_train/train_acc_test.jsonl  \
        --model_name_or_path    \
        --output_path   ../data/RAG_train/train_acc.jsonl \
        --cut_chunk  \
        --number_chunk ${items[$i]}  > "${items[$i]}".out  2>&1 &
done