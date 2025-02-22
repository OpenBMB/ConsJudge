import json
import re
import argparse
import random
import openai
import torch
import time
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


client = OpenAI(api_key="api key",base_url="url")  # mom key
data_list = []
resopnse_data = []



input_jsonlpath = 'Your evaluation analysis jsonl format file' #The file paths of the Marco and Wow datasets after postprocessing


def extract_answer(text):

    match = re.search(r"Average: (\w)", text)
    if match:
        return match.group(1)
    return None


def random_swap(list_a, list_b):

    l_a = copy.deepcopy(list_a)
    l_b = copy.deepcopy(list_b)
    for i in range(len(list_a)):
        if random.random() > 0.5:
            l_a[i], l_b[i] = l_b[i], l_a[i]
    return l_a, l_b


def extract_overall_score(text):

    match = re.search(r"Average:\s*([\d.]+)", text, re.DOTALL)
    if match:
        return float(match.group(1))
    return 0


def preprocess_text(text):

    return text.replace('\n', ' ')


def load_question_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['query'])
    return data


def load_ground_truth_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['answer'])
    return data


def load_output_jsonl(file_path):
    """Load JSONL file and return list of query values."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line_dict = json.loads(line)

            data.append(line_dict['output'])
    return data




question_data = load_question_jsonl(input_jsonlpath)
ground_truth_data = load_ground_truth_jsonl(input_jsonlpath)
output_data = load_output_jsonl(input_jsonlpath)


overall_score = 0


def generate_prompt(question_data, ground_truth_data, output_data):
    finresult1 = []
    prompt1 = []
    for i in range(500):
        if output_data[i] is not None:

            text1 = f"You are a professional evaluator. Your task is to evaluate the quality of the given text based on query and the ground truth from Accuracy,Informativeness and Fluency dimension.  Accuracy:1 star means Completely wrong,2 stars means Partially correct,3 stars means Completely correct;Informativeness:1 star means Lack of necessary information or off-topic response,2 stars means Insufficient information to provide a complete answer,3 stars means Information is correct and sufficient;Fluency:1 star means Many grammar errors,2 stars means Generally fluent but with some grammar errors,3 stars means Language is fluent and in accordance with human conventions;Logicality: 1 star means Confused and full of logical flaws,2 stars means Some logical issues present,3 stars means Logically sound.Harmlessness: 1 star means Violates ethics or is objectionable,2 stars means Mostly compliant but with a few flaws,3 stars means Fully compliant with public morality.Please give the star for each dimension and the final average star,such as 'Accuracy: a number,Informativeness: a number,Fluency: a number,Logicality:a number,Harmlessness:a number.Average: a number'.and you only output the result without any other explanation. query:{question_data[i]}\nground truth: {ground_truth_data[i]}\ntext: {output_data[i]}\nResult:"
            prompt1.append(text1)

    start_time = time.time()
    for index in range(500):

        text1 = prompt1[index]
        try:
            response = client.chat.completions.create(
                model="", # model name, nny model can be used
                messages=[
                    {"role": "user",
                     "content": text1[:11000]},
                ],
                max_tokens=64,
            )
            result1 = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")  
            result1 = 'none'  

        print(result1)
        print(extract_overall_score(result1))
        finresult1.append(extract_overall_score(result1))

        elapsed_time = time.time() - start_time
        print(f"Processed batch:{index}, took {elapsed_time:.2f} seconds")

    return finresult1


finresult = generate_prompt(question_data, ground_truth_data, output_data)

count = 0
for index in range(500):
    result = finresult[index]

    if result is not None:
        overall_score += (result / 3)
        count += 1


print(f"average score:", overall_score / count)