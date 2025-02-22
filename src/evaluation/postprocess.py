import json


def extract_and_merge(jsonl_file1, jsonl_file2, output_file):
    merged_data = []
    with open(jsonl_file1, 'r', encoding='utf - 8') as file1, open(jsonl_file2, 'r', encoding='utf - 8') as file2:
        for line1, line2 in zip(file1, file2):
            data1 = json.loads(line1)
            data2 = json.loads(line2)

            new_entry = {}

            if 'hotpotqa' in jsonl_file1:
                for key in ['input', 'output', 'metric_result', 'prompts', 'answer']:
                    if key in data1:
                        new_entry[key] = data1[key]
                if 'output' in data2:
                    new_entry['answer'] = data2['output'][0]['answer']
            elif 'tqa' in jsonl_file1:
                for key in ['input', 'output', 'metric_result', 'prompts']:
                    new_entry[key] = data1[key]
                    if 'output' in data2:
                        answer = []
                        for item in data2['output']:
                            if 'answer' in item:
                                answer.append(item['answer'])
                        new_entry['answer'] = answer
            elif 'nq' in jsonl_file1:
                for key in ['input', 'output', 'metric_result', 'prompts', 'answer']:
                    if key in data1:
                        new_entry[key] = data1[key]
                if 'output' in data2:
                    answer = []
                    for item in data2['output']:
                        if 'answer' in item:
                            answer.append(item['answer'])
                    new_entry[key] = answer
            elif 'trex' in jsonl_file1:
                for key in ['input', 'output', 'metric_result', 'prompts']:
                    new_entry[key] = data1[key]
                    if 'output' in data2:
                        answer = []
                        for item in data2['output']:
                            if 'answer' in item:
                                answer.append(item['answer'])
                        new_entry['answer'] = answer
            if 'marco' in jsonl_file1:
                for key in ['query', 'output', 'metric_result', 'prompts']:
                    new_entry[key] = data1[key]
                new_entry['answer'] = data1['answers'][0]

            if 'wow' in jsonl_file1:
                for key in ['output', 'metric_result', 'prompts']:
                    new_entry[key] = data1[key]
                new_entry['query'] = data1['input']
                if 'output' in data2:
                    new_entry['answer'] = data2['output'][0]['answer']
            elif 'asqa' in jsonl_file1:
                for key in ['question', 'output', 'metric_result', 'prompts', 'answer']:
                    new_entry[key] = data1[key]
                if 'qa_pairs' in data2:
                    short_answer = []
                    for qa_pair in data2['qa_pairs']:
                        short_answer.append(qa_pair['short_answers'])
                    new_entry['short_answers'] = short_answer

            merged_data.append(new_entry)

    # 将合并后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in merged_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    jsonl_file1 = 'Your evaluation output jsonl format file' #
    jsonl_file2 = 'Your evaluation input jsonl format file'
    output_file = 'Your evaluation analysis jsonl format file'
    extract_and_merge(jsonl_file1, jsonl_file2, output_file)