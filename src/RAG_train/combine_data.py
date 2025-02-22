import json
import random
from collections import defaultdict

random.seed(42)


def read_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        jsonl_data = []
        for line in file:
            json_obj = json.loads(line.strip())
            jsonl_data.append(json_obj)
    return jsonl_data


def save_list_to_jsonl(data_type, out_put_path, sub_list):
    file_path = out_put_path + '/' + data_type + '.jsonl'
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in sub_list:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')


raw_data = ''
dpo_data = ''

raw_data_list = read_data(raw_data)
dpo_data_list = read_data(dpo_data)

exist_key = ["aqua_rat", "ecqa", 'web_questions', 'wiki_qa', 'yahoo_answers_qa', "marcoqa", "strategyqa"]

for raw, dpo in zip(raw_data_list, dpo_data_list):
    raw_response = []
    aug_response = []
    for dd in dpo['context']:
        if dd['type'] == 'raw':
            raw_response.append(dd)
        if dd['type'] == 'aug_1-5':
            aug_response.append(dd)

    raw['raw_response'] = random.sample(raw_response, 2)
    raw['aug_response'] = random.sample(aug_response, 2)
    raw['passage'] = raw['rerank_passage']

grouped_dict = defaultdict(list)


key = 'data_type'


for item in raw_data_list:
    grouped_dict[item[key]].append(item)


grouped_list = list(grouped_dict.values())

out_put_path = ''

print("-------------------")
for sub_list in grouped_list:
    data_type = sub_list[0]['data_type']
    if data_type in exist_key:
        save_list_to_jsonl(data_type, out_put_path, sub_list)

print("---------------------")