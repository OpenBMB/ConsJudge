gpu_vis=0,1,2,3
MASTER_PORT=2357
nohup deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT ../src/RAG_train/train.py \
    --model_name_or_path  \
    --train_data_path ../data/RAG_train/train.jsonl \
    --eval_data_path  ../data/RAG_train/dev.jsonl \
    --max_length 1500 \
    --max_prompt_length 1400 \
    --output_dir  \
    --eval_steps 200 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir  \
    --bf16 True \
    --use_lora True \
    --num_train_epochs 1 \
    --top_n 5 \
    --deepspeed ../src/RAG_train/ds_config_zero2.json > run.log 2>&1 &