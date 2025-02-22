import json
import numpy as np
import string
import re
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_presence(short_answers, context):


    n_short_answers = [normalize_answer(sa) for sa in short_answers]

    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def compute_str_em(output,short_answers):


    loc_acc = []
    for short_answer in short_answers:
        loc_acc.append(exact_presence(short_answer,output))
    return np.mean(loc_acc)
   
def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def process_jsonl(file_path):

    total_score = 0
    count = 0


    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            output = data.get('output', '')
            short_answers = data.get('short_answers', [])
            
            if output and short_answers:
                score = compute_str_em(output, short_answers)
                total_score += score
                count += 1

    print(total_score/count)


file_path = 'Your asqa evaluation analysis jsonl format file'  #The file paths of the ASQA dataset after postprocessing
process_jsonl(file_path)
