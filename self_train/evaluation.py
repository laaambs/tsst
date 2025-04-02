from dataclasses import dataclass
import logging
import os
from dataset.clevr_math import ClevrDataset
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
from self_train.eval_beam import BEAM_Json_Evaluator
from utils.verify_correct import call_gpt

import yaml
import argparse
import datetime


multiple_choice_prompt = """You are tasked with answering a multiple-choice question based on the given image and question. Select the correct answer from the options (A, B, C, ...). Respond only with the letter of the correct answer.

Question: {question}

Options:
{options}

Your answer:
"""

qa_prompt = "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer with a digit. \n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\n"


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


beam_sample_prompt_no_image = "Answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"


# beam_sample_prompt = "Task: Given the image, answer the following question. \nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\""
class MLLM_evaluator:
    MULTIPLE_CHOICE = 'multiple_choice'
    MULTIPLE_CHOICE_STEPS = 'multiple_choice_steps'
    MULTIPLE_CHOICE_BEAM_SAMPLE = 'multiple_choice_steps_beam_sample'
    QA = 'qa'
    VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if two responses convey the same meaning and correctness.
- If Response 1 matches the correct answer in meaning and accuracy, reply 'True'
- If Response 1 contradicts the correct answer, reply 'False'
- If you cannot confidently determine the relationship, reply 'Unknown'
Remember to ONLY reply with True/False/Unknown."""
    SCQA_VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if the provided response match the correct answer for the given question.
    - If provided response matches the correct answer in meaning and accuracy, reply 'True'
    - If provided response contradicts the correct answer, reply 'False'
    - If you cannot confidently determine the relationship, reply 'Unknown'
    Remember to ONLY reply with True/False/Unknown.
    """
    
    def __init__(self, model, tokenizer, dataset, value_model, device, batch_size, task, image_processor,  processor: LlaVaProcessor, num_beams=1, branch=3, n_consistency=3):
        self.model = model
        self.value_model = value_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.branch = branch
        # TODO custom_llava_collate_fn by task
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.custom_llava_collate_fn)
        self.tokenizer = tokenizer
        self.device = device
        self.task = task
        self.task_type = self.get_task_type()
        self.prompt_template = self.get_prompt_template()
        print(f'Using prompt template: {self.prompt_template}')
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
    
    def evaluate_by_task(self, save_file, logger):
        if 'multiple_choice' in self.task or 'qa' in self.task:
            return self.evaluate_multiple_choice(save_file, logger)
        else:
            raise ValueError(f"Task {self.task} not supported")
        
    def evaluate_multiple_choice(self, save_file, logger):
        if isinstance(self.model, str):
            pass
        else:
            self.model.eval()
        results = []
        total_num = 0
        
        # 打开文件，追加写入
        with open(save_file, 'a', encoding='utf-8') as f:
            for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Processing batches"):
                total_num += 1
                # 获取模型的响应
                
                if isinstance(self.model, str):
                    if 'gpt' in self.model:
                        responses = call_gpt_by_batch(self.model, batch['img_path'], batch['formatted_questions'])
                else:
                    responses = self.processor.batch_inference(batch)
                if batch_idx == 0: 
                    # 仅打印第一批的响应
                    print(f'responses: {responses}')
                
                # 验证批次结果
                # verify_results = self.verify_by_batch(responses, batch)
                if type(self.dataset) == ScienceQADataset:
                    verify_results = self.verify_by_batch_sqa(responses, batch)
                else:
                    verify_results = self.verify_by_batch(responses, batch)
                    
                if batch_idx == 0:
                    print(f'verify_results: {verify_results}')
                
                # 收集所有结果
                results.extend(verify_results)
                
                # 对每个样本，构建字典并写入文件
                for i in range(len(responses)):
                    response = responses[i]
                    verify_result = verify_results[i]
                    sample = {
                        'data_id': batch['id'][i],
                        'question': batch['question'][i],
                        'answer_choices': batch['answer_choices'][i],
                        'answer_label': batch['answer_label'][i],
                        'answer_str': batch['answer_str'][i],
                        'response': response,
                        'verify_result': verify_result,
                        'image_path': batch['img_path'][i]
                    }
                    
                    # 写入文件
                    f.write(json.dumps(sample) + '\n')
                
                # 每个批次完成后记录日志
                logger.info(f"Processed batch {batch_idx + 1}/{len(self.dataloader)}")
        
        # 计算准确率
        acc = sum([1 if i else 0 for i in results]) / len(results)
        
        # 记录最终的准确率
        logger.info(f"Accuracy: {acc:.4f}")
        
        return acc
    
    
    def get_prompt_template(self):
        if self.task_type == self.MULTIPLE_CHOICE:
            return multiple_choice_prompt
        elif self.task_type == self.MULTIPLE_CHOICE_STEPS:
            return multiple_choice_steps_prompt
        elif self.task_type == self.MULTIPLE_CHOICE_BEAM_SAMPLE:
            return beam_sample_prompt
        elif self.task_type == self.QA:
            return qa_prompt
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
            print(f'response: {response}')
            answer_str = answer_strs[i]
            print(f'answer_str: {answer_str}')
            # print('='*20)
            # print(f'label: {label}')
            # print(f'response: {response}')
            res = self.verify_by_regular_expression(response, label)
            if res is None:
                res = self.verify_by_llm(response, answer_str)
            batch_result.append(res)
        return batch_result

    def verify_by_batch_sqa(self, responses, batch):
        batch_result = []
        labels = batch[self.label_name]
        # TODO
        # answer_strs = batch['answer_str']
        questions = batch['question']
        choices = batch['answer_choices']
        # labels = batch['answer_label']
        for i in range(len(responses)):
            label = labels[i]
            response = responses[i]
            # answer_str = answer_strs[i]
            question = questions[i]
            choice = choices[i]

            res = self.scqa_verify(response, question, choice, label)
            batch_result.append(res)
        return batch_result

    
    def verify_by_regular_expression(self, response, answer_label):
        print('verify by regular expression')
        if self.task_type in [self.MULTIPLE_CHOICE_STEPS, self.MULTIPLE_CHOICE_BEAM_SAMPLE, self.MULTIPLE_CHOICE]:
            choice = self.extract_model_answer_ABCD(response)
            if choice is None:
                return None
            choice_index = [i for i in string.ascii_uppercase].index(choice) + 1
            res = answer_label==choice_index
        elif self.task_type == self.QA:
            ans = self.extract_model_answer_digit(response)
            if ans is None:
                return None
            res = str(ans) == str(answer_label)
        print(f'verify res: {res}')
        return res
    
    def scqa_verify(self, response, question, choices, label):
        # label_letter = self.generate_label_string(label)
        # res = self.verify_by_regular_expression(response, label)
        # if res is None:
        choices_str = self.generate_options_string(choices)
        label_str = self.generate_label_string(label)
        answer_str = f"{label_str} {choices[label-1]}"
        res = self.scqa_verify_by_llm(response, question, choices_str, answer_str)
        return res
    
    def get_final_answer(self, response: str) -> str:
        match = re.search(r'final answer:\s*(.+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return self.extract_last_sentence(response)
        
    def build_scqa_comparison_prompt(self, response: str, question: str,
                                     choices: str, answer: str) -> str:
        prompt_template = """Here is a given multiple-choice question and its correct answer:\nQuestion: {question}\nOptions:\n{choices}\nCorrect Answer: {answer}\n\nHere is the response you need to evaluate:\nProvided Response: {response}\nIs Provided Response correct? Reply with only True/False/Unknown.
    True = Provided Response is correct
    False = Provided Response is incorrect
    Unknown = Cannot determine

    Answer with ONLY True/False/Unknown."""
        
        return prompt_template.format(response=response, question=question, choices=choices, answer=answer)
    
    def scqa_verify_by_llm(self, response, question, choices_str, answer_str: str):
        print('verify by llm')
        final_answer = self.get_final_answer(response)
        print(f'final_answer: {final_answer}')
        args=self.llm_verify_args
        print(f'answer_label str: {answer_str}')
        prompt = self.build_scqa_comparison_prompt(final_answer, question, choices_str, answer_str)
        # print(f'prompt: {prompt}')
        messages = [
            {"role": "system", "content": self.SCQA_VERIFY_SYSTEM_PROMPT},
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
    
    def extract_last_sentence(self, text):
        # 清理文本中的多余空白符
        text = text.strip()
        
        # 使用正则表达式查找所有的句子，以 . ? ! ; 为分隔符，包含空格后再匹配
        sentences = re.split(r'(?<=[.?!;])\s+', text)
        
        # 从最后一句开始检查，直到找到长度合适的句子
        for sentence in reversed(sentences):
            if len(sentence.strip()) >= 10:
                return sentence.strip()
        
        # 如果没有找到合适的句子，则返回最接近的一句话
        return sentences[-1].strip()
    
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
        elif 'qa' in self.task:
            return 'answer_label'
        else:
            raise NotImplementedError(f'Task not supported: {self.task}')

    def extract_model_answer_digit(self, output_text):
        if self.task_type == self.QA:
            match = re.search(r"Final Answer:\s*(\d+)", output_text.strip())
            if match:
                return match.group(1)
            sentences = output_text.strip().split('\n')
            last_sentence = sentences[-1].strip()
            match = re.search(r"\b\d+\b", last_sentence)
        else:
            raise NotImplementedError(f'Task type not supported: {self.task_type}')
        if match:
            return match.group(0)
        return None

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
        elif 'qa' in self.task:
            return self.QA
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
        def format_question(question, answer_choices, img_path):
            if self.task_type in [self.MULTIPLE_CHOICE, self.MULTIPLE_CHOICE_STEPS, self.MULTIPLE_CHOICE_BEAM_SAMPLE]:
                options = self.generate_options_string(answer_choices)
                if img_path:
                    res = self.prompt_template.format(question=question, options=options)  
                else:
                    res = beam_sample_prompt_no_image.format(question=question, options=options)
            elif self.task_type == self.QA:
                res = qa_prompt.format(question=question)
            else:
                NotImplementedError(f'Task type not supported: {self.task_type}')
            return res
        
        questions = [format_question(item["question"], item['answer_choices'], item['img_path']) for item in batch]
        images = [item["img_path"] for item in batch]
        collated_batch = {
            key: [item[key] for item in batch] for key in batch[0].keys()
        }
        
        collated_batch['formatted_questions'] = questions
        collated_batch['images'] = images
        
        if not isinstance(self.model, str):
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
    
    def generate_label_string(self, label):
        # 获取字母序列 (A, B, C, ...)
        letters = string.ascii_uppercase
        # 检查是否足够选项字母
        if label > len(letters):
            raise ValueError("Label is too large for available letter options.")
        # 生成 "A. 选项内容" 的字符串列表
        return f"{(letters[label-1])}"
    
    def check_end_condition(self, text):
        end_tokens = [
            "###", "---", "final answer", "user", "human", ". . .", "</s>"
        ]
        max_length = 1000
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

    def beam_sample(self, save_file, logger, args=None):
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
            num_beams = self.num_beams if len(candidate_solutions) > 1 else (2*self.num_beams)
            while len(final_solutions)<self.n_consistency and count < max_count:
                extended_candidate_solutions = []
                # extended_candidate_solutions_with_empty_candidate_sequences = []

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
                        # extended_candidate_solutions.append(solution["value"])
                        # extended_candidate_solutions_with_empty_candidate_sequences.append({
                        #     'value': solution["value"] + ' ' + 'Final Answer: ',
                        #     'score': solution["score"]
                        #     })
                    else:
                        if solution["value"] is None:
                            for candidate_sequence in candidate_sequences:
                                extended_candidate_solutions.append(candidate_sequence)
                        else:
                            for candidate_sequence in candidate_sequences:
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
                        
                # candidate_solutions.extend(extended_candidate_solutions_with_empty_candidate_sequences)
                print(f"step {count}, candidate_solutions: {candidate_solutions}")
                print(f"step {count}, final_solutions: {final_solutions}")
                count+=1
                
                if len(candidate_solutions)==0:
                    candidate_solutions = [{"value":None,"score":0.0}]

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

# from openai import ChatCompletion


# openai.api_base = 'https://api.mixrai.com'
# openai.api_key = 'sk-OaPARAh625BaWFiT6f713705702c4cB59e0e7c7e6455D3F4'


def call_gpt_by_batch(model, image_paths: list, prompts: list):
    import openai

    res = []
    for i, q in enumerate(prompts):
        
        print(q)
        print(image_paths[i])

        success = False
        retries = 0
        while retries < 10 and not success:
            try:
                openai.api_base = 'https://xiaoai.plus/v1'
                openai.api_key = 'sk-LXza5AaFNvn7OHyIPniNJm9u5kfCLAW9pC1W75pTF5Mg8IgD'
                print(openai.api_base)
                print(openai.api_key)
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=build_message_with_image(q, [image_paths[i]]),
                    max_tokens=300,
                )
                res_content = response.choices[0].message.content
                res.append(res_content)
                success = True  # If the request succeeds, exit the loop
            except Exception as e:
                retries += 1
                print(f"Error: {e}, Retrying... ({retries}/10)")
                time.sleep(3)  # Wait for 3 seconds before retrying
                
        if not success:
            print(f"Failed after 10 retries, appending 'None' for prompt {q}")
            res.append("None")  # Append 'None' if all retries failed
            
    return res

def build_message_with_image(prompt, image_paths):
    base64_images = [get_base64_image(image_path) if image_path else '' for image_path in image_paths]
    
    have_image = True
    for i in base64_images:
        if not i:
            have_image = False
    
    if have_image:
        image_contents = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            for base64_image in base64_images
        ]
    else:
        image_contents = []
        
    content = [{"type": "text", "text": prompt}] + image_contents

    messages=[
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
        

def eval_by_model_path(model_path, args, dataset, save_file, logger, mm_use_im_start_end=True):
            
    # use GPT, TODO: better implementation
    if 'gpt' in model_path:
        
        model = model_path
        tokenizer = None
        llava_processor = None
        image_processor = None
        task = args.task
        
        evaluator = MLLM_evaluator(model, tokenizer, dataset, value_model=None,batch_size=args.batch_size, device='cuda', task=task, processor=llava_processor, image_processor=image_processor)
        res = evaluator.evaluate_by_task(save_file, logger)
        return res
    
        
    
    llava_model = LLAVA(model_path)
    model, tokenizer, image_processor = llava_model.model, llava_model.tokenizer, llava_model.image_processor

    assert image_processor is not None
    llava_processor = LlaVaProcessor(args=args, model=model, tokenizer=tokenizer, image_processor=image_processor, mm_use_im_start_end=mm_use_im_start_end)
    task = args.task
    # task_type = args.task_type
    evaluator = MLLM_evaluator(model, tokenizer, dataset, value_model=None,batch_size=args.batch_size, device='cuda', task=task, processor=llava_processor, image_processor=image_processor)
    res = evaluator.evaluate_by_task(save_file, logger)
    return res

def eval_beam_sample_by_model_path(model_path, value_model, args, dataset, save_file, verify_save_file, logger, mm_use_im_start_end=True):
    llava_model = LLAVA(model_path)
    
    model, tokenizer, image_processor = llava_model.model, llava_model.tokenizer, llava_model.image_processor

    assert image_processor is not None
    llava_processor = LlaVaProcessor(args=args, model=model, tokenizer=tokenizer, image_processor=image_processor, mm_use_im_start_end=mm_use_im_start_end)
    task = args.task
    assert args.batch_size == 1, 'Only support batch_size=1 now'
    evaluator = MLLM_evaluator(model, tokenizer, dataset, value_model=value_model, batch_size=args.batch_size, device='cuda', task=task, processor=llava_processor, image_processor=image_processor, num_beams=args.num_beams, branch=args.branch, n_consistency=args.n_consistency)
    
    res = evaluator.beam_sample(save_file, logger)
    evaluator = BEAM_Json_Evaluator(save_file)
    absolute_acc, voting_acc, greedy_acc = evaluator.all_verify_beam(logger, verify_save_file, dataset_name='scienceqa')
    print({
        'greedy_acc': greedy_acc,
        'absolute_acc': absolute_acc,
        'voting_acc': voting_acc,
    })
    
    return absolute_acc, voting_acc 



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










class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)  # Recursively convert sub-dictionaries
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return str(self._to_dict())

    def _to_dict(self):
        """Convert the Config object back to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value._to_dict()
            result[key] = value
        return result

class ConfigManager:
    def __init__(self, yaml_path, args):
        self.yaml_path = yaml_path
        self.args = args
        self.config = self._load_yaml_config()
        self._merge_args_with_config()

    def _load_yaml_config(self):
        """Load configuration from a YAML file."""
        with open(self.yaml_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def _merge_args_with_config(self):
        """Merge command-line arguments into the YAML configuration."""
        for key, value in vars(self.args).items():
            if value is not None:  # Override YAML config only if argument is provided
                self.config[key] = value

    def get_config(self):
        """Return the merged configuration as an object supporting dot access."""
        return self.config
    
    def get_all_args(self):
        return Config(self.config)

# Define the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Example script using YAML and args.")
    parser.add_argument('--yaml_path', type=str, default='config/eval/scienceqa_direct.yaml', help="Path to the YAML config file.")
    return parser.parse_args()


def setup_logging(log_file):
    logger = logging.getLogger(log_file)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Example usage
def parse_args_form_yaml():
    args = parse_args()
    config_manager = ConfigManager(args.yaml_path, args)
    args = config_manager.get_all_args()
    print('==========config==========')
    print(args)

    return args


if __name__ == "__main__":
    
    current_directory = os.getcwd()
    
    args = parse_args_form_yaml()
    
    
    start_time = time.time()
    
    dataset_name = args.dataset_name
    dataset_root = args.dataset_root
    data_subset = args.data_subset
    data_partition = args.data_partition
    sample_size = args.sample_size
    seed = args.seed
    # sample_size = args.sample_size
    # prepare dataset
    if 'vcr' in dataset_name:
        dataset = VCRDataset(dataset_root, dataset_name, data_subset, data_partition, caption_path=None)
        
    elif 'solution' in dataset_name or 'scienceqa' in dataset_name:
        dataset = ScienceQADataset(dataset_root, dataset_name, data_partition=data_partition, seed=seed)
    elif 'clevr' in dataset_name:
        jsonl_file = "sampled_validation.jsonl"
        dataset = ClevrDataset(jsonl_file)
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported.")

    # prepare model
    models = args.models
    
    for model in models:
        if not model['run_flag']:
            continue
        
        print(f"Processing model: {model}")
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        
        tag = model['tag']
        policy_path = model['policy_model']
        use_value = model['use_value']
        sample_strategy = 'beam' if use_value else 'direct'
        exp_tag = f'{tag}_{sample_strategy}_{sample_size}_{time_str}'
        
        logging_file = os.path.join(current_directory, f'logs/eval/{args.dataset_name}/{exp_tag}.log')
        os.makedirs(os.path.dirname(logging_file), exist_ok=True)

        logger = setup_logging(logging_file)
        logger.info(args)
        logger.info(f"Processing model: {model}")

        save_dataset_name=args.dataset_name.replace('/data1/taowei/datasets/ScienceQA/', '_')
        save_file = os.path.join(current_directory, f'output/eval_data/{save_dataset_name}_{exp_tag}.jsonl')
            
        if use_value:
            value_path = model['value_model']
            value_name = get_model_name_from_path(value_path)
            tokenizer, value_model, _, _ = load_pretrained_model(
                value_path, None, value_name
            )
            verify_save_file = os.path.join(current_directory, f'output/verified_data/{save_dataset_name}_{exp_tag}_verify.jsonl')
            eval_beam_sample_by_model_path(policy_path, value_model, args, dataset, save_file, verify_save_file, logger, mm_use_im_start_end=True)
        else:
            acc = eval_by_model_path(policy_path, args, dataset, save_file, logger, mm_use_im_start_end=True)
            print(f"Accuracy: {acc}")
            logger.info(f"Accuracy: {acc}")

    end_time = time.time()
    print('END')
    print(f"Total time: {end_time - start_time}")