from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import json
import re
import torch
import argparse
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser(description="Run LLM generation with specified parameters.")
parser.add_argument("--input_jsonlpath", type=str, required=True, help="Path to input JSONL file.")
parser.add_argument("--output_jsonlpath", type=str, required=True, help="Path to output JSONL file.")
parser.add_argument("--model_path", type=str, required=True, help="model path.")
args = parser.parse_args()

input_jsonlpath = args.input_jsonlpath
output_jsonlpath = args.output_jsonlpath
model_path = args.model_path

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=768)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, gpu_memory_utilization=0.3, max_model_len=3000,
          trust_remote_code=True, tensor_parallel_size=1)

data_list = []
response_data1 = []
response_data2 = []
response_data3 = []
response_data4 = []
response_data5 = []
response_data6 = []
response_data7 = []
response_data8 = []



def load_question_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['question'])
    return data


def load_choiceA_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['choiceA'])
    return data


def load_choiceB_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['choiceB'])
    return data


def load_choiceC_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['choiceC'])
    return data


def load_choiceD_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['choiceD'])
    return data


def load_answer_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['answer'])
    return data

def extract_information(sentences):

    extracted_sentences = []


    for sentence in sentences:

        cot_match = re.search(r"COT:\s*(.+?)\s*Answer:", sentence, re.DOTALL | re.IGNORECASE)
        if not cot_match:
            return

        cot_content = cot_match.group(1)
        cot_content = re.sub(r"\\n|\\|\n", " ", cot_content).strip()


        best_answer_match = re.search(r"Best\s*answer:\s*([A-Za-z])", sentence, re.IGNORECASE)
        worst_answer_match = re.search(r"Worst\s*answer:\s*([A-Za-z])", sentence, re.IGNORECASE)

        if not best_answer_match or not worst_answer_match:
            return

        best_answer = best_answer_match.group(1)

        worst_answer = worst_answer_match.group(1)

        if best_answer == worst_answer or best_answer not in ['A', 'B', 'C', 'D'] or worst_answer not in ['A', 'B', 'C','D']:
            return
        extracted_sentences.append(
            f"[COT]:{cot_content}[Answer]:Best answer:{best_answer}.Worst answer:{worst_answer}."
        )

    return extracted_sentences


question_data = load_question_jsonl(input_jsonlpath)
answer_data = load_answer_jsonl(input_jsonlpath)
choiceA_data = load_choiceA_jsonl(input_jsonlpath)
choiceB_data = load_choiceB_jsonl(input_jsonlpath)
choiceC_data = load_choiceC_jsonl(input_jsonlpath)
choiceD_data = load_choiceD_jsonl(input_jsonlpath)

choiceA = []
choiceB = []
choiceC = []
choiceD = []

# Prepare your prompts
prompts1 = []
prompts2 = []
prompts3 = []
prompts4 = []
prompts5 = []
prompts6 = []
prompts7 = []
prompts8 = []

for index in range(len(question_data)):
    arr = [choiceA_data[index], choiceB_data[index], choiceC_data[index], choiceD_data[index]]
    random.shuffle(arr)
    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Hallucination aspect.hallucination refers to the presence of information in the option that contradicts ground truth,it is an incorrect answer to the question.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts1.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Completeness aspect.completeness refers to whether the choice contains as complete information as possible from the ground truth,it did not fully answer the question correctly.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts2.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Coherence aspect.coherence refers to whether the choice is logically coherent and whether the language between each sentence is fluent.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts3.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Semantic consistency aspect.semantic consistency refers to whether the choice is semantically consistent with the ground truth, rather than just having lexical repetition.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts4.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Hallucination,Completeness aspects.hallucination refers to the presence of information in the option that contradicts ground truth,it is an incorrect answer to the question.completeness refers to whether the choice contains as complete information as possible from the ground truth,it did not fully answer the question correctly.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts5.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Coherence,Semantic consistency aspects.coherence refers to whether the choice is logically coherent and whether the language between each sentence is fluent.semantic consistency refers to whether the choice is semantically consistent with the ground truth, rather than just having lexical repetition.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts6.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Hallucination,Completeness,Semantic consistency aspects.hallucination refers to the presence of information in the option that contradicts ground truth,it is an incorrect answer to the question.completeness refers to whether the choice contains as complete information as possible from the ground truth,it did not fully answer the question correctly.semantic consistency refers to whether the choice is semantically consistent with the ground truth, rather than just having lexical repetition.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts7.append(text)

    text = (
        f"you are a excellent evaluation expert.Please select the best answer and the worst answer from four options based on the ground truth and the query from Hallucination,Completeness,Coherence,Semantic consistency aspects.hallucination refers to the presence of information in the option that contradicts ground truth,it is an incorrect answer to the question.completeness refers to whether the choice contains as complete information as possible from the ground truth,it did not fully answer the question correctly.coherence refers to whether the choice is logically coherent and whether the language between each sentence is fluent.semantic consistency refers to whether the choice is semantically consistent with the ground truth, rather than just having lexical repetition.Note:your result format must strictly be'COT:{{there is your analysis,as detailed as possible}}.Answer:Best answer:{{a choice must be one of[A,B,C,D]}}.Worst answer:{{a choice must be one of[A,B,C,D]}}'.Output the content of COT first and then output the Answer.Your analysis must be concise and you must first output the Best answer and then output the Worst answer in the Answer.When you output Best answer and Worst answer, you can only output one selection from [A, B, C, D]\nHere is the query:{question_data[index]}\nHere is the ground truth:{answer_data[index]}\nHere is the A choice:{arr[0]}\nHere is the B choice:{arr[1]}\nHere is the C choice:{arr[2]}\nHere is the D choice:{arr[3]}\nresult:")
    prompts8.append(text)

    choiceA.append(arr[0])
    choiceB.append(arr[1])
    choiceC.append(arr[2])
    choiceD.append(arr[3])

# Prepare the prompts for the model
texts1 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts1]

# generate outputs
outputs1 = llm.generate(texts1, sampling_params)

# Print the outputs.
for output in outputs1:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data1.append(f"{generated_text!r}")

texts2 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts2]

# generate outputs
outputs2 = llm.generate(texts2, sampling_params)

# Print the outputs.
for output in outputs2:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data2.append(f"{generated_text!r}")

texts3 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts3]

# generate outputs
outputs3 = llm.generate(texts3, sampling_params)

# Print the outputs.
for output in outputs3:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data3.append(f"{generated_text!r}")

texts4 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts4]

# generate outputs
outputs4 = llm.generate(texts4, sampling_params)

# Print the outputs.
for output in outputs4:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data4.append(f"{generated_text!r}")

texts5 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts5]

# generate outputs
outputs5 = llm.generate(texts5, sampling_params)

# Print the outputs.
for output in outputs5:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data5.append(f"{generated_text!r}")

texts6 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts6]

# generate outputs
outputs6 = llm.generate(texts6, sampling_params)

# Print the outputs.
for output in outputs6:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data6.append(f"{generated_text!r}")

texts7 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts7]

# generate outputs
outputs7 = llm.generate(texts7, sampling_params)

# Print the outputs.
for output in outputs7:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data7.append(f"{generated_text!r}")

texts8 = [tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in prompts8]

# generate outputs
outputs8 = llm.generate(texts8, sampling_params)

# Print the outputs.
for output in outputs8:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    response_data8.append(f"{generated_text!r}")

with open(output_jsonlpath, "a", encoding="utf8") as output_file:

    for index in range(len(question_data)):
        question = question_data[index]

        answer = answer_data[index]
        choiceA_result = choiceA[index]
        choiceB_result = choiceB[index]
        choiceC_result = choiceC[index]
        choiceD_result = choiceD[index]
        response1 = response_data1[index]
        response2 = response_data2[index]
        response3 = response_data3[index]
        response4 = response_data4[index]
        response5 = response_data5[index]
        response6 = response_data6[index]
        response7 = response_data7[index]
        response8 = response_data8[index]
        cots = [response1,response2,response3,response4,response5,response6,response7,response8]
        extrect_cots = extract_information(cots)

        if extrect_cots is not None:
            data_list.append({"question": question, 'answer': answer, 'choiceA': choiceA_result, 'choiceB': choiceB_result,
                              'choiceC': choiceC_result, 'choiceD': choiceD_result, 'cot':[extrect_cots[0],extrect_cots[1],extrect_cots[2],extrect_cots[3],extrect_cots[4],extrect_cots[5],extrect_cots[6],extrect_cots[7]]})
    data_str_list = [json.dumps(data) for data in data_list]
    for data_str in data_str_list:
        output_file.write(data_str + '\n')