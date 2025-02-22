import json
import os
import random

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_subfolders(path):

    all_items = os.listdir(path)

    sub_folders = [item for item in all_items if os.path.isdir(os.path.join(path, item))]

    return sub_folders

def random_element(array):
    return random.choice(array)

def get_files_in_path(path):
    files_array = []
    for root, dirs, files in os.walk(path):
        for file in files:
            files_array.append(os.path.join(root, file))
    return files_array

def save_checkpoint(model, tokenizer, accelerator, args, output_dir):
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir, safe_serialization=False)
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict, safe_serialization=False
        )


