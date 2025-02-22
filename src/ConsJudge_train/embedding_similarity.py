from tqdm import tqdm
import torch
import argparse
import os
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Run LLM generation with specified parameters.")
parser.add_argument("--input_jsonlpath", type=str, required=True, help="Path to input JSONL file.")
parser.add_argument("--output_jsonlpath", type=str, required=True, help="Path to output JSONL file.")
parser.add_argument("--model_path", type=str, required=True, help="model path.")
args = parser.parse_args()

input_jsonlpath = args.input_jsonlpath
output_jsonlpath = args.output_jsonlpath
model_path = args.model_path

model = SentenceTransformer(model_path, trust_remote_code=True,
                            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16})


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_answer(content):

    if "[Answer]:" in content:
        return content.split("[Answer]:", 1)[1].strip()
    return None

def load_jsonl(file_path, key):
    """Load JSONL file and return list of specific key values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_dict = json.loads(line)
            data.append(line_dict[key])
    return data


cot_list = load_jsonl(input_jsonlpath, 'cot')
question_data = load_jsonl(input_jsonlpath, 'question')
answer_data = load_jsonl(input_jsonlpath, 'answer')
choiceA_data = load_jsonl(input_jsonlpath, 'choiceA')
choiceB_data = load_jsonl(input_jsonlpath, 'choiceB')
choiceC_data = load_jsonl(input_jsonlpath, 'choiceC')
choiceD_data = load_jsonl(input_jsonlpath, 'choiceD')

chosen_data = []
rejected_data = []
data_list = []


with tqdm(total=len(cot_list), desc="Processing COT Data") as pbar:
    for cot in cot_list:

        sentence_embeddings = model.encode(cot).tolist()


        sentence_embeddings_tensor = torch.tensor(sentence_embeddings).cuda()
        similarity_matrix = torch.matmul(sentence_embeddings_tensor, sentence_embeddings_tensor.T)


        row_sums = torch.sum(similarity_matrix, 1)
        row_means = row_sums / similarity_matrix.size(1)
        row_means = row_means.cpu().numpy()


        max_index = np.argmax(row_means)
        min_index = np.argmin(row_means)


        chosen_cot = cot[max_index]
        rejected_cot = cot[min_index]


        if row_means[max_index] - row_means[min_index] >= 0.05:
            chosen_data.append(chosen_cot)
            rejected_data.append(rejected_cot)
        else:
            chosen_data.append("")
            rejected_data.append("")


        pbar.update(1)

temp_data = []  # 存储符合条件的数据
for index in range(len(question_data)):
    question = question_data[index]
    answer = answer_data[index]
    choiceA_result = choiceA_data[index]
    choiceB_result = choiceB_data[index]
    choiceC_result = choiceC_data[index]
    choiceD_result = choiceD_data[index]
    chosen = chosen_data[index]
    rejected = rejected_data[index]
    chosen_answer = extract_answer(chosen)
    rejected_answer = extract_answer(rejected)

    if chosen != "" and rejected != "" and chosen_answer != rejected_answer:
        temp_data.append({
            "question": question,
            "answer": answer,
            "choiceA": choiceA_result,
            "choiceB": choiceB_result,
            "choiceC": choiceC_result,
            "choiceD": choiceD_result,
            "chosen": chosen,
            "rejected": rejected,
        })

# 根据answer长度分成两部分
short_answer_data = [item for item in temp_data if len(item['answer']) < 20]
long_answer_data = [item for item in temp_data if len(item['answer']) >= 20]

# 分别筛选出10000条小于30的答案和12000条大于30的答案
short_answer_data = short_answer_data[:8500]
print(len(short_answer_data))
long_answer_data = long_answer_data[:13500]
print(len(long_answer_data))

# 合并数据
final_data = short_answer_data + long_answer_data

transformed = []
with open(output_jsonlpath, "a", encoding="utf8") as output_file:
    with tqdm(total=len(final_data), desc="Writing Output Data") as pbar:
        for item in final_data:
            question = item['question']
            answer = item['answer']
            choiceA_result = item['choiceA']
            choiceB_result = item['choiceB']
            choiceC_result = item['choiceC']
            choiceD_result = item['choiceD']
            chosen = item['chosen']
            rejected = item['rejected']

            conversation_value = (
                f"you are a excellent evaluation expert.Please select the appropriate dimension from Hallucination, Completeness, Coherence, and Semantic consistency based on the ground truth and thr query to choose the best answer and worst answer from the four choices.hallucination refers to the presence of information in the choice that contradicts ground truth,it is an incorrect answer to the question.completeness refers to whether the choice contains as complete information as possible from the ground truth,it did not fully answer the question correctly.Coherence refers to whether the choice is logically coherent and whether the language between each sentence is fluent.semantic consistency refers to whether the choice is semantically consistent with the ground truth, rather than just having lexical repetition.Note:your result format must strictly be'[COT]:{{there is your analysis}}[Answer]:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}.'Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you must only output one choice from [A, B, C, D].\nHere is the query:{question}\nHere is the ground truth:{answer}\nHere is the A choice:{choiceA_result}\nHere is the B choice:{choiceB_result}\nHere is the C choice:{choiceC_result}\nHere is the D choice:{choiceD_result}\nresult:"
            )
            transformed.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": conversation_value
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": chosen
                },
                "rejected": {
                    "from": "gpt",
                    "value": rejected
                }
            })
            pbar.update(1)

save_json(transformed, output_jsonlpath)

