import json
import torch
from torch.utils.data import Dataset
import os
import random
import re
import os
from tqdm import tqdm
import string
import logging



process_solution_template = """ The following is a question-response pair. Please revise the given response into a step-by-step reasoning process. Try to use declarative sentences in each reasoning step to describe the answering process and refrain from using imperative sentences. Do not include any content that does not appear in the question and the existing response. Be as concise as possible.

Question:
{hint_question}

Choices:
{choices}
Correct choice: {answer_str}

Existing Response:
{caption}

Your revised response should follow this format:
Here is the step-by-step reasoning process:
Step 1: ...
Step 2: ...
...
Step *: ...
Final Answer: [the correct choice content] """

class ScienceQADataset(Dataset):
    def __init__(self, dataset_root, dataset_name, data_partition, seed):
        
        if "val" in dataset_name:
            sub_dir = "val"
        elif "train" in dataset_name:
            sub_dir = "train"
        else:
            raise ValueError("dataset_name should be one of ['train', 'val']")

        print(f"Loading {sub_dir} data from {dataset_root}...")
        
        jsonl_file = os.path.join(dataset_root, dataset_name)
        
        if data_partition:
            start, end = map(int, data_partition.split("_"))
            if start < 0 or end < start:
                raise ValueError("Invalid data_partition range.")
            
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
        
        self.total_length = len(data)
        
        if end >= self.total_length:
            raise ValueError(
                f"data_partition out of range. Available range is 0_{self.total_length - 1}."
            )
        # if seed is not None:
        #     random.seed(seed)
        #     random.shuffle(data)
            
        self.data = data[start:end + 1]
        
        self.img_dir = dataset_root


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img_path = data.get("img_path", None)
        if img_path is not None:
            img_path = os.path.join(self.img_dir, img_path)
        else:
            img_path = None
        hint = data.get("hint", None)
        if hint is not None:
            question = hint + " " + data["question"]
        else:
            question = data["question"]
        return {
            "id": data["id"],
            "question": question,
            "question_with_choices": "",
            "answer_choices": data["answer_choices"],
            "answer_str": data["answer_str"],
            "img_path": img_path,
            "meta_path": "",
            "processed_img_path": "",
            "answer_label": data.get("answer_label"),
            "caption": data.get("caption"),
        }

def setup_logging(log_file):
    logger = logging.getLogger(log_file)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def generate_options_string(choices):
    letters = string.ascii_uppercase
    if len(choices) > len(letters):
        raise ValueError("Too many choices for available letter options.")
    options = [f"({letters[i]}) {choice};" for i, choice in enumerate(choices)]
    # 将选项按行拼接为字符串
    return "\n".join(options)

def build_process_prompt(hint, question, choices, caption, answer_str):
    if hint == "":
        hint_question = question
    else:
        hint_question = hint + "\n" + question
    choice_str = generate_options_string(choices)
    prompt = process_solution_template.format(hint_question=hint_question.strip(), 
                                              choices=choice_str.strip(),
                                              answer_str=answer_str.strip(),
                                              caption=caption.strip())
    return prompt
