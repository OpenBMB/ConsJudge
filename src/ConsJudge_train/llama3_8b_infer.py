import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser(description="Run LLM generation with specified parameters.")
parser.add_argument("--temperature", type=float, required=True, help="Temperature for sampling.")
parser.add_argument("--input_jsonlpath", type=str, required=True, help="Path to input JSONL file.")
parser.add_argument("--output_jsonlpath", type=str, required=True, help="Path to output JSONL file.")
parser.add_argument("--model_path", type=str, required=True, help="model path.")

args = parser.parse_args()


temperature = args.temperature
input_jsonlpath = args.input_jsonlpath
output_jsonlpath = args.output_jsonlpath
model_path = args.model_path

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(n=1, temperature=temperature, top_p=0.8, repetition_penalty=1.05, max_tokens=256)
llm = LLM(model=model_path, gpu_memory_utilization=0.6,
          trust_remote_code=True, tensor_parallel_size=1)


def load_jsonl(file_path, key):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_dict = json.loads(line)
            data.append(line_dict[key])
    return data

question_data = load_jsonl(input_jsonlpath, 'question')
content_data = load_jsonl(input_jsonlpath, 'content')
answer_data = load_jsonl(input_jsonlpath, 'answer')

prompts = []

for index in range(len(question_data)):
    if content_data[index] != "":
        text = (f"You are a knowledgeable expert,please think carefully and answer directly the given question based on the given content.Note:{{Answer directly the given question,return a quit precise answer}} \
                            Here is the question:{question_data[index]} \
                            Here is the content:{content_data[index]}")
        prompts.append(text)
    else:
        text = (f"You are a knowledgeable expert,please think carefully and answer directly the given question.Note:{{Answer directly the given question,return a quit precise answer}} \
                           Here is the question:{question_data[index]}")
        prompts.append(text)

texts = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts]


outputs = llm.generate(texts, sampling_params)

data_list = []
responses = [[] for _ in range(1)]
for output in outputs:
    for i in range(1):
        if i < len(output.outputs):
            responses[i].append(output.outputs[i].text)
        else:
            responses[i].append("")

with open(output_jsonlpath, "a", encoding="utf8") as output_file:
    for index in range(len(question_data)):
        question = question_data[index]
        answer = answer_data[index]
        response_data = {f"llama3-8b-answer{i+1}": responses[i][index] for i in range(1)}
        record = {"question": question, "answer": answer, **response_data}
        output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
