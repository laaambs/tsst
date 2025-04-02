from dataclasses import dataclass
import logging
from models.LLaVA.llava_inference import LLAVA, LlaVaProcessor
from dataset.vcr import VCRDataset
from dataset.science_qa import ScienceQADataset
from torch.utils.data import DataLoader
import torch
from transformers import StoppingCriteria

from ast import List
from typing import Union, List
from models.LLaVA.llava.model.builder import load_pretrained_model
from models.LLaVA.llava.mm_utils import get_model_name_from_path
import torch
import argparse
from models.LLaVA.llava.conversation import (SeparatorStyle,
                                        conv_templates)
from models.LLaVA.llava.mm_utils import KeywordsStoppingCriteria
import time
from tqdm import tqdm
import string
import random
from models.LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.LLaVA.llava.conversation import conv_templates
from models.LLaVA.llava.model.builder import load_pretrained_model
from models.LLaVA.llava.utils import disable_torch_init
from models.LLaVA.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)

from PIL import Image

import json

import requests
from PIL import Image
from io import BytesIO
import re
from utils.verify_correct import call_gpt
import numpy as np


multiple_choice_prompt = """You are tasked with answering a multiple-choice question based on the given image and question. Select the correct answer from the options (A, B, C, ...). Respond only with the letter of the correct answer.

Question: {question}

Options:
{options}

Your answer:
"""


multiple_choice_steps_prompt = [
    "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.",

    "In the \"Final Answer\", you should only respond with the letter of the correct choice.",
    "Here is the question and options:",
    "Question: {question}",
    "Options:",
    "{options}",
    # "A. {option_A}\n"
    # "B. {option_B}\n"
    # "C. {option_C}\n"
    # "D. {option_D}\n"
    "Your response must follow this format:",
    "\"Here is the step-by-step reasoning process:",
    "Step 1: ...",
    "Step 2: ...",
    "Step n: ...",
    "Final Answer: ...\"",
    "Your answer:"
]
    
multiple_choice_steps_prompt = '\n'.join(multiple_choice_steps_prompt)

beam_sample_prompt = "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"

# beam_sample_prompt = "Task: Given the image, answer the following question. \nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\""
class MLLM_evaluator:
    MULTIPLE_CHOICE = 'multiple_choice'
    MULTIPLE_CHOICE_STEPS = 'multiple_choice_steps'
    MULTIPLE_CHOICE_BEAM_SAMPLE = 'multiple_choice_steps_beam_sample'
    VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if two responses convey the same meaning and correctness.
- If Response 1 matches the correct answer in meaning and accuracy, reply 'True'
- If Response 1 contradicts the correct answer, reply 'False'
- If you cannot confidently determine the relationship, reply 'Unknown'
Remember to ONLY reply with True/False/Unknown."""
    
    def __init__(self, model, tokenizer, dataset, value_model, device, batch_size, task, image_processor,  processor: LlaVaProcessor, num_beams=1, branch=3, n_consistency=3):
        self.model = model
        self.value_model = value_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.branch = branch
        # TODO custom_llava_collate_fn by task
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.custom_llava_collate_fn)
        self.tokenizer = tokenizer
        self.device = device
        self.task = task
        self.task_type = self.get_task_type()
        self.prompt_template = self.get_prompt_template()
        self.label_name = self.get_label_name_by_task()
        self.processor = processor
        self.image_processor = image_processor
        self.n_consistency = n_consistency
        print(f'task type: {self.task_type}')
        self.llm_verify_args = {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "temp_llm": 0.0,  
            "max_new_tokens": 1024,
            "config": {
                "openai_api_base": "https://api.deepinfra.com/v1/openai",
                "openai_keys": {
                    "KYgozM7pEA7KKhUvs9AJ0rRrPiuQbakp": 0, 
                    "qHU7Lr4JOQWlIHt7vmqAH6UEI3eRW8La": 0,
                    # "SOfjAUp6gagi1dyRL2X3dcWPvduZjpMj": 0,
                    # "jUPdi6SUwH3ftqzOv2yHWrbH9uL258vk": 0,
                    "qi0JfbGb5fQem8HN1nVOIrKxdZFo3iGT": 0,
                    "1zDZ5ljkWl5vXafOiuGzA09vS05Rh8BE": 0,
                }
            }
        }
        
    # def evaluate(self):
    #     self.model.eval()
    #     total_loss = 0
    #     total_samples = 0
    #     with torch.no_grad():
    #         for batch in self.data_loader:
    #             batch = {k: v.to(self.device) for k, v in batch.items()}
    #             outputs = self.model(**batch)
    #             loss = outputs.loss
    #             total_loss += loss.item()
    #             total_samples += len(batch['input_ids'])
                
    #     return total_loss / total_samples
    
    def evaluate_by_task(self):
        if 'multiple_choice' in self.task:
            return self.evaluate_multiple_choice()
        else:
            raise ValueError(f"Task {self.task} not supported")
        
    def evaluate_multiple_choice(self):
        self.model.eval()
        results = []
        total_num = 0
        
        # write to jsonl file
        # with open(f'llava_scienceqa_1000_train_results.jsonl', 'a') as f:
            
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Processing batches"):
            total_num += 1
            # print(f"Batch {batch_idx + 1}:")
            responses = self.processor.batch_inference(batch)
            if batch_idx == 0: 
                # print(f'batch: {batch}')
                print(f'responses: {responses}')
            verify_results = self.verify_by_batch(responses, batch)
            if batch_idx == 0:
                print(f'verify_results: {verify_results}')
            results.extend(verify_results)
            for i in range(len(responses)):
                response = responses[i]
                verify_result = verify_results[i]
                science_qa_sample = {
                    'data_id': batch['id'][i],
                    'question': batch['question'][i],
                    'answer_choices': batch['answer_choices'][i],
                    'answer_label': batch['answer_label'][i],
                    'answer_str': batch['answer_str'][i],
                    'response': response,
                    'verify_result': verify_result,
                    'image_path': batch['img_path'][i]
                }
                    # f.write(json.dumps(science_qa_sample) + '\n')
        acc = sum([1 if i else 0 for i in results]) / len(results)
        return acc
    
    
    def get_prompt_template(self):
        if self.task_type == self.MULTIPLE_CHOICE:
            return multiple_choice_prompt
        elif self.task_type == self.MULTIPLE_CHOICE_STEPS:
            return multiple_choice_steps_prompt
        elif self.task_type == self.MULTIPLE_CHOICE_BEAM_SAMPLE:
            return beam_sample_prompt
        else:
            raise NotImplementedError(f'Task type not supported: {self.task_type}')

    
    def verify_by_batch(self, responses, batch):
        batch_result = []
        labels = batch[self.label_name]
        # TODO
        answer_strs = batch['answer_str']
        for i in range(len(responses)):
            label = labels[i]
            response = responses[i]
            answer_str = answer_strs[i]
            # print('='*20)
            # print(f'label: {label}')
            # print(f'response: {response}')
            res = self.verify_by_regular_expression(response, label)
            if res is None:
                res = self.verify_by_llm(response, answer_str)
            batch_result.append(res)
        return batch_result

    
    def verify_by_regular_expression(self, response, answer_label):
        print('verify by regular expression')
        choice = self.extract_model_answer_ABCD(response)
        if choice is None:
            return None
        choice_index = [i for i in string.ascii_uppercase].index(choice) + 1
        res = answer_label==choice_index
        print(f'verify res: {res}')
        return res
    
    def verify_by_llm(self, response, answer_label: str):
        print('verify by llm')
        match = re.search(r'final answer:\s*(.+)', response, re.IGNORECASE)
        final_answer = match.group(1).strip() if match else response
        print(f'final_answer: {final_answer}')
        args=self.llm_verify_args
        print(f'answer_label str: {answer_label}')
        prompt = self.build_comparison_prompt(final_answer, answer_label)
        messages = [
            {"role": "system", "content": self.VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        gpt_response, _ = call_gpt(
            messages,
            model=args["model"],
            temp_gpt=args["temp_llm"],
            max_new_tokens=args["max_new_tokens"],
            config=args["config"],
        )
        response = gpt_response.strip().lower()
        print(f'gpt_response: {response}')
        print('='*20)
        
        match = re.search(r'\b(true|false|unknown)\b', response, re.IGNORECASE)
        res = match.group(0).lower() if match else None
        print(f'verify res: {res}')
        return res == 'true'
    
    def get_label_name_by_task(self):
        if 'multiple_choice' in self.task:
            return 'answer_label'
        else:
            raise NotImplementedError(f'Task not supported: {self.task}')


    def extract_model_answer_ABCD(self, output_text):
        if self.task_type in [self.MULTIPLE_CHOICE_STEPS, self.MULTIPLE_CHOICE_BEAM_SAMPLE]:
            match = re.search(r"Final Answer:\s*(A|B|C|D)", output_text.strip())
            if match:
                return match.group(1)
            sentences = output_text.strip().split('\n')
            last_sentence = sentences[-1].strip()
            match = re.search(r"\b(A|B|C|D)\b", last_sentence)
            
        elif self.task_type == self.MULTIPLE_CHOICE:
            # 定义正则表达式，匹配单独的一行 A, B, C, D
            match = re.search(r"\b(A|B|C|D)\b", output_text.strip())            
        else:
            raise NotImplementedError(f'Task type not supported: {self.task_type}')
        if match:
            return match.group(1)
        return None
    
    def image_parser(self, image_file, sep=","):
        out = image_file.split(sep)
        return out
    
    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    def get_task_type(self):
        if 'multiple_choice' in self.task:
            if 'steps' in self.task:
                if 'beam' in self.task:
                    return self.MULTIPLE_CHOICE_BEAM_SAMPLE
                else:
                    return self.MULTIPLE_CHOICE_STEPS
            else:
                return self.MULTIPLE_CHOICE
        else:
            raise NotImplementedError(f'Task not supported: {self.task}')


    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out
    
    
    def custom_llava_collate_fn(self, batch):
        """
        自定义 collate_fn，将 batch 中的问题和图像路径处理为 input_ids 和 image_tensors。
        """
        # 过滤掉 None 数据
        batch = [b for b in batch if b is not None]

        # input_ids_list = []
        # image_tensors_list = []
        def format_question(question, answer_choices):
            if self.task_type in [self.MULTIPLE_CHOICE, self.MULTIPLE_CHOICE_STEPS, self.MULTIPLE_CHOICE_BEAM_SAMPLE]:
                options = self.generate_options_string(answer_choices)
                res = self.prompt_template.format(question=question, options=options)            
            else:
                NotImplementedError(f'Task type not supported: {self.task_type}')
            return res
        questions = [format_question(item["question"], item['answer_choices']) for item in batch]
        images = [item["img_path"] for item in batch]
        collated_batch = {
            key: [item[key] for item in batch] for key in batch[0].keys()
        }
        collated_batch['formatted_questions'] = questions
        batch_image_tensor, batch_input_ids = self.processor.get_processed_tokens_batch(questions, images)
        collated_batch["input_ids"] = batch_input_ids
        collated_batch["image_tensors"] = batch_image_tensor

        return collated_batch
    
    def build_comparison_prompt(self, response: str, answer: str) -> str:
        prompt_template = """Compare these two responses to the same question:
    Response 1 (to be verified): {response}
    Response 2 (correct answer): {answer}

    Is Response 1 correct? Reply with only True/False/Unknown.
    True = Response 1 is correct
    False = Response 1 is incorrect
    Unknown = Cannot determine

    Answer with ONLY True/False/Unknown."""
        
        return prompt_template.format(response=response, answer=answer)
    
    def generate_options_string(self, choices):
        # 获取字母序列 (A, B, C, ...)
        letters = string.ascii_uppercase
        # 检查是否足够选项字母
        if len(choices) > len(letters):
            raise ValueError("Too many choices for available letter options.")
        # 生成 "A. 选项内容" 的字符串列表
        options = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]
        # 将选项按行拼接为字符串
        return "\n".join(options)

    # def topk_beam_sample_by_batch(self):
    #     self.model.eval()
        
    #     final_solutions = [[] for _ in range(self.batch_size)]
    #     solutions = [[] for _ in range(self.batch_size)]
        
    #     for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Processing batches"):
            
            
    #         solution_index = 0
    #         while any(len(final_solutions[i]) < self.n_consistency for i in range(self.batch_size)):
    #             batch_text = []
    #             # Step 1: Generate candidate solutions
    #             if all(not solutions[i] for i in range(self.batch_size)):
    #                 batch_text = batch["formatted_questions"]
    #                 candidate_sequence_sets = self.processor.batch_inference_beam(batch, self.num_beams)
    #             else:
    #                 batch_text = []
    #                 batch_image_paths = []
    #                 active_indices = []

    #                 for i in range(self.batch_size):
    #                     for solution in solutions[i]:
    #                         batch_text.append(solution.value)
    #                         batch_image_paths.append(solution.image_path)
    #                         active_indices.append(i)

    #                 candidate_sequence_sets = [set() for _ in range(self.batch_size)]
    #                 for start in range(0, len(batch_text), self.batch_size):
    #                     end = min(start + self.batch_size, len(batch_text))
    #                     sub_batch_text = batch_text[start:end]
    #                     sub_batch_image_paths = batch_image_paths[start:end]

    #                     batch_image_tensor, batch_input_ids = self.processor.get_processed_tokens_batch(sub_batch_text, sub_batch_image_paths)
    #                     sub_batch = {"input_ids": batch_input_ids, "image_tensors": batch_image_tensor}
    #                     sub_candidate_sets = self.processor.batch_inference_beam(sub_batch, self.num_beams)
    #                     candidate_sequence_sets.extend(sub_candidate_sets)

    #             extended_candidate_solutions = [[] for _ in range(self.batch_size)]

            
    #             # Step 2: Extend solutions with generated candidates
    #             for i, candidate_set in enumerate(candidate_sequence_sets):
    #                 for candidate in candidate_set:
    #                     text = batch_text[i]
    #                     new_value = f"{text} {candidate}".strip()
    #                     new_solution = Solution(
    #                             value=new_value,
    #                             # score=solution.score,  # Temporary score before updating
    #                             # batch_idx=i,
    #                             image_path=solution.image_path,
    #                             # parent=solution,
    #                             end_condition=self.check_end_condition(new_value)  # Modify based on actual ending condition check
    #                         )
    #                     extended_candidate_solutions[i].append(new_solution)
                        
    #                     # for solution in solutions[i] or [self.format_solution(batch_idx=i, image_path=batch[i]["img_path"])]:
    #                     #     new_value = f"{text} {candidate}".strip()
    #                     #     new_solution = Solution(
    #                     #         value=new_value,
    #                     #         score=solution.score,  # Temporary score before updating
    #                     #         # batch_idx=i,
    #                     #         image_path=solution.image_path,
    #                     #         parent=solution,
    #                     #         end_condition=self.check_end_condition(new_value)  # Modify based on actual ending condition check
    #                     #     )
    #                     #     extended_candidate_solutions[i].append(new_solution)
                            
    #             # Step 3: Update scores using value model
    #             for i in range(self.batch_size):
    #                 # TODO
    #                 scores = self.processor.evaluate_solutions(extended_candidate_solutions[i])
    #                 for solution, score in zip(extended_candidate_solutions[i], scores):
    #                     solution.score = score
                        
    #             # Step 4: Sort by score
    #             for i in range(self.batch_size):
    #                 extended_candidate_solutions[i].sort(key=lambda sol: sol.score, reverse=True)

    #             # Step 5: Prune by branch size
    #             for i in range(self.batch_size):
    #                 extended_candidate_solutions[i] = extended_candidate_solutions[i][:self.branch]
                    
    #             # Step 6: Check for ending solutions and finalize
    #             for i in range(self.batch_size):
    #                 remaining_solutions = []
    #                 for solution in extended_candidate_solutions[i]:
    #                     if solution.end_condition:  # Replace with actual end condition logic
    #                         final_solutions[i].append(solution)
    #                     else:
    #                         remaining_solutions.append(solution)
    #                 extended_candidate_solutions[i] = remaining_solutions
                    
    #             solutions = extended_candidate_solutions

    
    # def update_solutions(self, candidata_solutions, solutions, final_solutions):
    #     for i in range(len(solutions)): #batch_size
    #         solution = solutions[i]
    #         # check end condition
    #         end_condition = self.check_end_condition(solution.value)
    #         if end_condition:
    #             solution.end_condition = True
    #             final_solution = final_solutions[i]
    #             final_solution.append(solution)
    #         # add to solutions
    #         solutions[i] = solution
            

    def check_end_condition(self, text):
        end_tokens = [
            "###", "---", "final answer", "user", "human", ". . .", "</s>"
        ]
        max_length = 1200
        end_pattern = re.compile("|".join(map(re.escape, end_tokens)), re.IGNORECASE)
        return bool(end_pattern.search(text)) or len(text) > max_length
    
    # def format_solution(self, value=None, score=0.0, batch_idx=None, image_path="", parent=None, end_condition=False):
    #     return Solution(value=value, score=score, batch_idx=batch_idx, image_path=image_path, parent=parent, end_condition=end_condition)
    
    # def update_solutions_by_batch_candidate_sequence_sets(self, candidate_sequence_sets, solutions):
    #     for i in range(len(candidate_sequence_sets)):
    #         # len(candidate_sequence_sets) should be batch_size
    #         candidate_sequence_set = candidate_sequence_sets[i]
    #         solution = solutions[i]
    #         solution = self.update_solution(solution, candidate_sequence_set)
    def beam_sample(self, save_file, logger):
        # value_model = self.value_model
        
        all_samples = []

        stop_words_ids=torch.tensor([[29889]]).to(self.device)
        
        assert self.batch_size == 1 , "batch_size should be 1"
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processing sampling:")):
            logger.info(f"Process Sample {batch_idx}")
            candidate_solutions = [{"value":None,"score":0.0}]
            final_solutions = []
            query = batch["formatted_questions"][0]
            print(f"query: {query}")
            images_tensor = batch["image_tensors"].to(self.device, dtype=torch.float16)
            
            
            count = 0
            max_count = 30
            min_len = 3
            num_beams = self.num_beams if len(candidate_solutions) > 1 else (2*self.num_beams)
            while len(final_solutions)<self.n_consistency and count < max_count:
                extended_candidate_solutions = []

                for solution in candidate_solutions:
                    prompt = self.processor.prepare_prompt_beam_sampling(query=query, solution=solution["value"])
            
                    input_ids = (
                            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                            .unsqueeze(0)
                            .cuda()
                        )
                
                    stopping_criteria = BEAMKeywordsStoppingCriteria(tokenizer=self.tokenizer, stops=stop_words_ids, batch_size=input_ids.shape[0] * num_beams)

                    with torch.inference_mode():
                        output = self.model.generate(
                            input_ids,
                            images=images_tensor,
                            # image_sizes=image_sizes,
                            do_sample=True if self.processor.args.temperature > 0 else False,
                            temperature=self.processor.args.temperature,
                            max_new_tokens=256,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            early_stopping=True, 
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            return_dict_in_generate=True,
                        )
                
                    candidate_sequences = stopping_criteria.stopped_sequences
                    print(f"step {count}, candidate_sequences: {candidate_sequences}")
                    
                    if not candidate_sequences and solution["value"] is not None:
                        # solution_value = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
                        # value_prompt = self.processor.prepare_prompt_beam_sampling(query, extended_candidate_solution)
                        # solution_score = self.get_score(value_prompt, images_tensor)
                        # solution = {
                        #     'value': solution_value,
                        #     'score': solution_score
                        # }
                        final_solutions.append(solution)
                        continue
                        # final_solutions.append(solution)
                        # continue
                        extended_candidate_solutions.append(solution["value"])
                    else:
                        if solution["value"] is None:
                            for candidate_sequence in candidate_sequences:
                                if len(candidate_sequence.strip()) >= min_len:
                                    extended_candidate_solutions.append(candidate_sequence)
                        else:
                            for candidate_sequence in candidate_sequences:
                                if len(candidate_sequence.strip()) >= min_len:
                                    extended_candidate_solutions.append(solution["value"] + " " + candidate_sequence)
                
                candidate_scores = [0] * len(extended_candidate_solutions)
                
                # TODO: value的推理改为batch inference
                for i, extended_candidate_solution in enumerate(extended_candidate_solutions):
                    value_prompt = self.processor.prepare_prompt_beam_sampling(query, extended_candidate_solution)
                    candidate_score = self.get_score(value_prompt, images_tensor)
                    candidate_scores[i] = candidate_score.item()
                    
                candidate_solutions = []
                for value, score in zip(extended_candidate_solutions, candidate_scores):
                    candidate_solutions.append({
                        'value': value,
                        'score': score
                    })
                
                candidate_solutions = sorted(candidate_solutions, key=lambda x: x['score'], reverse=True)
                
                print(f"step {count}, candidate_solutions: {candidate_solutions}")
                
                # for solution in candidate_solutions[:]:
                #     if self.check_end_condition(solution['value']):
                #         final_solutions.append(solution)
                #         candidate_solutions.remove(solution)
                
                if len(candidate_solutions)>self.branch:
                    # candidate_solutions = candidate_solutions[:self.branch]
                    # total_score = sum(solution['score'] for solution in candidate_solutions)
                    # probabilities = [solution['score'] / total_score for solution in candidate_solutions]
                    temperature = 0.5
                    total_score = sum(pow(solution['score'], 1/temperature) for solution in candidate_solutions)
                    probabilities = [pow(solution['score'], 1/temperature) / total_score for solution in candidate_solutions]
                    candidate_solutions = random.choices(candidate_solutions, weights=probabilities, k=self.branch)
                
                for solution in candidate_solutions[:]:
                    if self.check_end_condition(solution['value']):
                        final_solutions.append(solution)
                        candidate_solutions.remove(solution)
                        
                print(f"step {count}, final_solutions: {final_solutions}")
                count+=1
                
                if len(candidate_solutions)==0:
                    candidate_solutions = [{"value":None,"score":0.0}]
                    
            if len(final_solutions)==0 and count >= max_count:
                if len(candidate_solutions) < self.n_consistency:
                    final_solutions.extend(candidate_solutions)
                else:
                    final_solutions.extend(candidate_solutions[:self.n_consistency])

            samples = sorted(final_solutions, key=lambda x: x['score'], reverse=True)
            # formatted_samples = [self.format_solution_sample(sample, batch) for sample in samples]
            formatted_samples = self.format_solution_samples(samples, batch)
            all_samples.append(formatted_samples)
            with open(save_file, "a") as jsonl_file:
                jsonl_file.write(json.dumps(formatted_samples) + "\n")
            logger.info(formatted_samples)
        return all_samples
    
    def get_score(self, value_prompt, images_tensor):
        value_input_ids = (
            tokenizer_image_token(value_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            candidate_score = self.value_model.generate(
                value_input_ids,
                images=images_tensor,
                # image_sizes=image_sizes,
            )
        return candidate_score
    
    def format_solution_sample(self, sample, batch):
        assert self.batch_size == 1
        return {
            "solution": sample['value'],
            "summary": '',
            "correct": False,
            "data_id": batch['id'][0],
            "image_path": batch['img_path'][0],
            "question": batch['question'][0],
            "data_id": ''
        }
        
    def format_solution_samples(self, samples, batch):
        assert self.batch_size == 1
        return {
            'data_id': batch['id'][0],
            'question': batch['question'][0],
            'answer_choices': batch['answer_choices'][0],
            'answer_label': batch['answer_label'][0],
            'answer_str': batch['answer_str'][0],
            'result': samples,
            "image_path": batch['img_path'][0],
        }



    # def 
# @dataclass
# class Solution:
#     value: str
#     score: float
#     batch_idx: int
#     image_path: str
#     parent: Union[None, 'Solution']
#     end_condition: bool
#     score: float
        
        

def eval_by_model_path(model_path, args, dataset, mm_use_im_start_end=True):
    llava_model = LLAVA(model_path)
    model, tokenizer, image_processor = llava_model.model, llava_model.tokenizer, llava_model.image_processor

    assert image_processor is not None
    llava_processor = LlaVaProcessor(args=args, model=model, tokenizer=tokenizer, image_processor=image_processor, mm_use_im_start_end=mm_use_im_start_end)
    task = args.task
    # task_type = args.task_type
    evaluator = MLLM_evaluator(model, tokenizer, dataset, value_model=None,batch_size=batch_size, device='cuda', task=task, processor=llava_processor, image_processor=image_processor)
    res = evaluator.evaluate_by_task()
    return res

def eval_beam_sample_by_model_path(model_path, value_model, args, dataset, save_file, logger, mm_use_im_start_end=True):
    llava_model = LLAVA(model_path)
    
    model, tokenizer, image_processor = llava_model.model, llava_model.tokenizer, llava_model.image_processor

    assert image_processor is not None
    llava_processor = LlaVaProcessor(args=args, model=model, tokenizer=tokenizer, image_processor=image_processor, mm_use_im_start_end=mm_use_im_start_end)
    task = args.task
    # task_type = args.task_type
    evaluator = MLLM_evaluator(model, tokenizer, dataset, value_model=value_model,batch_size=batch_size, device='cuda', task=task, processor=llava_processor, image_processor=image_processor, num_beams=args.num_beams, branch=args.branch, n_consistency=args.n_consistency)
    
    res = evaluator.beam_sample(save_file, logger)
    
    return res



class BEAMKeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops = [], batch_size = 1):
        super().__init__() 
        self.tokenizer = tokenizer
        self.stops = stops
        self.stopped_sequences = set()
        self.stop_flags = torch.zeros(batch_size , dtype=torch.bool, device=stops.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.stop_flags.all():
            return True
        
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            if self.stop_flags[batch_idx]:
                continue
            
            current_ids = input_ids[batch_idx]
            for stop in self.stops:
                if stop.shape[0] == 1:
                    matches = torch.where(current_ids == stop.item())[0]
                    if len(matches) > 0:
                        stop_index = matches[0].item() + 1
                        self._process_stop(current_ids, stop_index, batch_idx)
                        break
                else:
                    for i in range(len(current_ids) - len(stop) + 1):
                        if torch.equal(current_ids[i:i+len(stop)], stop):
                            self._process_stop(current_ids, i+len(stop), batch_idx)
                            break
        return bool(self.stop_flags.all().item()) 
    
    def _process_stop(self, input_ids, stop_index, batch_idx):
        if stop_index == 0:
            decoded_output = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            decoded_output = self.tokenizer.decode(input_ids[:stop_index], skip_special_tokens=True)
        self.stopped_sequences.add(decoded_output.strip())
        self.stop_flags[batch_idx] = True 

def setup_logging(log_file):
    logger = logging.getLogger(log_file)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    
    # calculate time
    start_time = time.time()
    
    
    parser = argparse.ArgumentParser(description="Evaluation script for model inference.")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum generation length.")
    parser.add_argument("--do_sample", type=bool, default=False, help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return.")
    parser.add_argument("--task", type=str, default='vcr', help="Task to evaluate.")
    parser.add_argument("--dataset_name", type=str, default='vcr_val', help="Dataset name.")
    parser.add_argument("--dataset_root", type=str, default='datasets/vcr1/', help="Dataset root.")
    parser.add_argument("--data_subset", type=str, default='datasets/vcr1/vcr_val_random500_annoid.yaml', help="Data subset.")
    parser.add_argument("--data_partition", type=str, default=None, help="Data partition.")
    parser.add_argument("--branch", type=int, default=3, help="Branch size.")
    parser.add_argument("--n_consistency", type=int, default=3, help="Number of consistent solutions.")
    # parser.add_argument("--task", type=str, default='multiple_', help="Task to evaluate.")

    
    args = parser.parse_args()
    print(args)
    
    dataset_root = args.dataset_root
    dataset_name = args.dataset_name
    data_subset = args.data_subset
    data_partition = args.data_partition
    batch_size = 4
    sample_size = 500
    seed = 42
    branch = args.branch
    
    if 'vcr' in args.dataset_name:
        dataset = VCRDataset(dataset_root, dataset_name, data_subset, data_partition, caption_path=None)
        
    elif 'scienceqa' in args.dataset_name:
        dataset = ScienceQADataset(dataset_root, dataset_name, sample_size=sample_size, seed=seed, data_partition=data_partition)
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not supported.")
    

    model_dict = {
        # "llava 1.5 7b": 'models/llava-v1.5-7b',
        # "llava rlhf 1.5 7b": 'models/llava-RLAIF-V-7B',
        # "sft policy 7b": "models/llava-v1.5-7b-sft-policy-v2",
        # "sft-policy-7b-vcr-epoch1": "models/llava-v1.5-7b-policy-v1-merge",
        # "llava-v1.5-7b-policy-v2-merge": "models/vcr/llava-v1.5-7b-policy-v2-merge",
        # "gen scienceqa": 'models/llava-v1.5-7b-sft-policy-v2'
        "ScienceQA v0 policy": "models/scienceqa/policy/llava-v1.5-7b-policy-v0"
        
    }
    
    # value_path = "models/llava-v1.5-7b-sft-prm-v5-best"
    # value_path = "models/llava-v1.5-7b-value-prm-v1-merge"
    use_value = False
    if use_value:
        value_path = 'models/llava-v1.5-7b-prm-v1-merge'
        model_base = None
        value_name = get_model_name_from_path(value_path)
        tokenizer, value_model, _, _ = load_pretrained_model(
            value_path, model_base, value_name
        )
    
    
    # value_model
    # def get_score(value_prompt, images_tensor):
    #     value_input_ids = (
    #         tokenizer_image_token(value_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    #         .unsqueeze(0)
    #         .cuda()
    #     )
    #     with torch.inference_mode():
    #         candidate_score = value_model.generate(
    #             value_input_ids,
    #             images=images_tensor,
    #             # image_sizes=image_sizes,
    #         )
    #     return candidate_score

    
    # print(get_score)
    
    print(f'dataset: {dataset_name}')
    results = {}
    for model in tqdm(model_dict, desc="Evaluating models"):
        model_path = model_dict[model]
        time_str = time.strftime("%Y%m%d-%H%M%S")
        logging_file = f'LLaVA-REST-MCTS/output/eval_data/{model}_{dataset_name}_beam_sample_{time_str}.log'
        logger = setup_logging(logging_file)

        save_file = f'LLaVA-REST-MCTS/output/eval_data/{model}_{dataset_name}_beam_sample_{time_str}.jsonl'
        # main eval
        if use_value:
            acc = eval_beam_sample_by_model_path(model_path=model_path, value_model=value_model, dataset=dataset, args=args, save_file=save_file, logger=logger)
        else:
            acc = eval_by_model_path(model_path=model_path, dataset=dataset, args=args)

        print('='*20)
        print(f'model name: {model}')
        print(f'model path: {model_path}')
        print(f'acc: {acc}')
        results[model] = acc
        
        
        # test beam sample
        # llava_model = LLAVA(model_path)
        # model, tokenizer, image_processor = llava_model.model, llava_model.tokenizer, llava_model.image_processor

        # assert image_processor is not None
        # llava_processor = LlaVaProcessor(args=args, model=model, tokenizer=tokenizer, image_processor=image_processor, mm_use_im_start_end=True)
        # task = args.task
        # # task_type = args.task_type
        # evaluator = MLLM_evaluator(model, tokenizer, dataset, value_model=value_model,batch_size=batch_size, device='cuda', task=task, processor=llava_processor, image_processor=image_processor, num_beams=args.num_beams, branch=branch)
        # res = evaluator.beam_sample()

    print(results)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script execution time: {elapsed_time:.4f} seconds")
        
    