import argparse
import torch
import json
import re
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


def split_list_evenly(lst, n):
    length = len(lst)
    avg = length // n
    remainder = length % n
    out = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        out.append(lst[start:end])
        start = end

    return out


# 提取 Best 和 Worst 答案的函数
def extract_answers(data):
    pattern_choice_best = r'Best\s*answer\s*:\s*\(?([A-D])\)?'
    pattern_choice_worst = r'Worst\s*answer\s*:\s*\(?([A-D])\)?'
    matches_best = re.findall(pattern_choice_best, data)
    matches_worst = re.findall(pattern_choice_worst, data)
    best = matches_best[0] if matches_best else 'E'
    worst = matches_worst[0] if matches_worst else 'E'
    return best, worst


# 读取 JSONL 文件内容为列表
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


# 写入 JSONL 文件
def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str,
                        default='',
                        help="The path of the training/evaluation data file to be processed.",
                        )
    parser.add_argument('--model_name_or_path', type=str,
                        default='t',
                        help="The path of the LLM.")
    parser.add_argument('--output_path', type=str,
                        default=None,
                        help="The path of the DPO data generated by LLM.")
    parser.add_argument('--cut_chunk', type=int,
                        default=8,
                        help="The number of data segments.",
                        )
    parser.add_argument('--number_chunk', type=int,
                        default=0,
                        help="The current index of data segments.",
                        )
    args = parser.parse_args()

    all_text_data = read_jsonl(args.input_data_path)
    split_input = split_list_evenly(all_text_data, args.cut_chunk)
    input_list = split_input[args.number_chunk]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=768)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1, dtype='bfloat16', trust_remote_code=True,
              gpu_memory_utilization=0.9, max_model_len=4096)

    prompts = []
    # 生成 prompts
    for text in input_list:
        prompt = f"""Please select the appropriate dimension from Hallucination, Completeness, Coherence, and Semantic consistency based on ground truth and query to choose the best answer and worst answer from the four options.hallucination refers to the presence of information in the answer that contradicts ground truth.\ncompleteness refers to whether the answer contains as complete information as possible from the ground truth.\ncoherence refers to whether the answer is logically coherent and whether the language between each sentence is fluent.\nsemantic consistency refers to whether the answer is semantically consistent with the ground truth, rather than just having lexical repetition.\nNote:{{you output format must strictly be'[cot]{{there is your analysis}}[answer]Best answer:{{your choice}}.Worst answer:{{your choice}}.'}}
        Here is the query: {text['question']}
        Here is the ground truth: {text['answer']}
        Here is the A choice: {text['raw_response'][0]['text']}
        Here is the B choice: {text['raw_response'][1]['text']}
        Here is the C choice: {text['aug_response'][0]['text']}
        Here is the D choice: {text['aug_response'][1]['text']}
        """

        prompts.append(prompt)

    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    ) for prompt in prompts]

    # 调用模型生成输出
    outputs = llm.generate(prompts, sampling_params)

    # 处理生成的输出并保存结果
    for index, (text, output) in enumerate(zip(input_list, outputs)):
        generated_text = output.outputs[0].text
        print(f"Generated text for {text['question']}: {generated_text}")

        best_answer, worst_answer = extract_answers(generated_text)
        print(best_answer)
        print(worst_answer)
        if best_answer == worst_answer or best_answer not in ['A', 'B', 'C', 'D'] or worst_answer not in ['A', 'B', 'C',
                                                                                                          'D']:
            continue
        else:
            # 找到对应的text元素，并更新chosen和rejected字段
            if best_answer == 'A':
                text['chosen'] = text['raw_response'][0]['text']
            elif best_answer == 'B':
                text['chosen'] = text['raw_response'][1]['text']
            elif best_answer == 'C':
                text['chosen'] = text['aug_response'][0]['text']
            elif best_answer == 'D':
                text['chosen'] = text['aug_response'][1]['text']

            if worst_answer == 'A':
                text['rejected'] = text['raw_response'][0]['text']
            elif worst_answer == 'B':
                text['rejected'] = text['raw_response'][1]['text']
            elif worst_answer == 'C':
                text['rejected'] = text['aug_response'][0]['text']
            elif worst_answer == 'D':
                text['rejected'] = text['aug_response'][1]['text']

            # 保存更新后的数据
            text['question'] = text['question']
            text['answer'] = text['answer']
            text['passage'] = text['passage'][:5]
            text['chosen'] = text.get('chosen', None)
            text['rejected'] = text.get('rejected', None)
            text['chosen'] = {"text": text['chosen']}
            text['rejected'] = {"text": text['rejected']}
    # 只保存chosen和rejected都不为空的条

    # 保存到新的 JSONL 文件
    #  保存到新的 JSONL 文件
    out_put_path = os.path.join(args.output_path, 'rouge.{}.jsonl'.format(args.number_chunk))
    with open(out_put_path, 'w', encoding='utf-8') as f:
        for item in input_list:
            # 确保chosen和rejected都不是None
            if item.get('chosen') is not None and item.get('rejected') is not None and item.get('chosen')[
                'text'] is not None and item.get('rejected')['text'] is not None:
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()