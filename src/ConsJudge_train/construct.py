import json
import random


def load_jsonl(file_path, keys):

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                line_dict = json.loads(line)
                filtered_data = {key: line_dict[key] for key in keys if key in line_dict}
                data.append(filtered_data)
            except json.JSONDecodeError:
                print(f"skip: {line}")
    return data

def create_new_jsonl(cpm_2b_files, cpm_4b_files, llama3_8b_files, qwen_14b_files, output_file_path):

    cpm_2b_data = [load_jsonl(file, ['question', 'answer','cpm-2b-answer1']) for file in cpm_2b_files]
    cpm_4b_data = [load_jsonl(file, ['cpm-4b-answer1']) for file in cpm_4b_files]
    llama3_8b_data = [load_jsonl(file, ['llama3-8b-answer1']) for file in llama3_8b_files]
    qwen_14b_data = [load_jsonl(file, ['qwen1.5-14b-answer1']) for file in qwen_14b_files]


    with open(output_file_path, 'w', encoding='utf-8') as output_file:

        for i in range(len(cpm_2b_data[0])):

            question = cpm_2b_data[0][i]['question']
            answer = cpm_2b_data[0][i]['answer']


            choiceA = random.choice([data[i]['cpm-2b-answer1'] for data in cpm_2b_data if len(data) > i])
            choiceB = random.choice([data[i]['cpm-4b-answer1'] for data in cpm_4b_data if len(data) > i])
            choiceC = random.choice([data[i]['llama3-8b-answer1'] for data in llama3_8b_data if len(data) > i])
            choiceD = random.choice([data[i]['qwen1.5-14b-answer1'] for data in qwen_14b_data if len(data) > i])


            record = {
                "question": question,
                "answer": answer,
                "choiceA": choiceA,
                "choiceB": choiceB,
                "choiceC": choiceC,
                "choiceD": choiceD
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"数据已保存到 {output_file_path}。")


cpm_2b_files = ['../data/ConsJudge_train/cpm_2b_0.5.jsonl', '../data/ConsJudge_train/cpm_2b_0.6.jsonl', '../data/ConsJudge_train/cpm_2b_0.7.jsonl']
cpm_4b_files = ['../data/ConsJudge_train/cpm_4b_0.5.jsonl', '../data/ConsJudge_train/cpm_4b_0.6.jsonl', '../data/ConsJudge_train/cpm_4b_0.7.jsonl']
llama3_8b_files = ['../data/ConsJudge_train/llama3_8b_0.5.jsonl', '../data/ConsJudge_train/llama3_8b_0.6.jsonl', '../data/ConsJudge_train/llama3_8b_0.7.jsonl']
qwen_14b_files = ['../data/ConsJudge_train/qwen_14b_0.5.jsonl', '../data/ConsJudge_train/qwen_14b_0.6.jsonl', '../data/ConsJudge_train/qwen_14b_0.7.jsonl']


output_file_path = '../data/ConsJudge_train/test_choices.jsonl'


create_new_jsonl(cpm_2b_files, cpm_4b_files, llama3_8b_files, qwen_14b_files, output_file_path)
