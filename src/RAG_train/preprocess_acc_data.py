import json
import random



def _acc_score(prediction, ground_truth):
    if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
        return 1.0
    else:
        return 0.0



def extract_cot_before(prediction):

    cot_pos = prediction.find('[COT]')
    if cot_pos != -1:
        return prediction[:cot_pos].strip()
    else:
        return prediction.strip()



def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]



def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')



def process_data(input_file, output_file):
    all_text_data = read_jsonl(input_file)
    processed_data = []
    valid_data_count = 0

    for text in all_text_data:

        raw_responses = [item['text'] for item in text['raw_response']]
        aug_responses = [item['text'] for item in text['aug_response']]


        answer = text['answer']


        positive = [extract_cot_before(response) for response in raw_responses + aug_responses if
                    _acc_score(extract_cot_before(response), answer) == 1.0]
        negative = [extract_cot_before(response) for response in raw_responses + aug_responses if
                    _acc_score(extract_cot_before(response), answer) == 0.0]


        if len(positive) < 4:
            if positive:
                positive.extend(random.choices(positive, k=4 - len(positive)))
            else:
                continue
        if len(negative) < 4:
            if negative:
                negative.extend(random.choices(negative, k=4 - len(negative)))
            else:
                continue


        passage = text['passage'][:5]


        text['positive'] = [{"text": response} for response in positive[:4]]  # 保证数组长度为 4
        text['negative'] = [{"text": response} for response in negative[:4]]  # 保证数组长度为 4
        text['passage'] = passage


        processed_data.append(text)
        valid_data_count += 1


    print(f"valid data numbers: {valid_data_count}")


    write_jsonl(output_file, processed_data)



input_file = ""  # The path for processing previous files.
output_file = ""  # The path of the processed file
process_data(input_file, output_file)