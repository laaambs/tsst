import json
import jsonlines
from tqdm import tqdm

from typing import Dict, Any
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)  # for exponential backoff
import requests
from requests.exceptions import Timeout
import time
import logging
import uuid
import os
SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if two responses convey the same meaning and correctness.
- If Response 1 matches the correct answer in meaning and accuracy, reply 'True'
- If Response 1 contradicts the correct answer, reply 'False'
- If you cannot confidently determine the relationship, reply 'Unknown'
Remember to ONLY reply with True/False/Unknown."""

beam_sample_prompt = "<image>\nGiven the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"

beam_sample_prompt_no_image = "Answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"

class APIKeyManager:
    def __init__(self, keys_dict):
        self.keys = keys_dict

    def get_available_key(self, max_retries=5, retry_delay=1):
        for attempt in range(max_retries):
            for key, status in self.keys.items():
                if status == 0:
                    self.keys[key] = 1
                    return key

            if attempt < max_retries - 1:
                print(f"没有可用的API密钥，等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)

        raise ValueError(f"在{max_retries}次尝试后仍没有可用的API密钥")

    def release_key(self, key):
        self.keys[key] = 0


key_manager = None


def set_openai_config(config):
    global key_manager
    openai.api_base = config["openai_api_base"]
    key_manager = APIKeyManager(config["openai_keys"])


@retry(
    wait=wait_random_exponential(min=0.1, max=0.2),
    stop=stop_after_attempt(10),
    retry=(
        retry_if_exception_type(Timeout) | retry_if_exception_type(openai.error.Timeout)
    ),
)
def call_gpt(chatgpt_messages, model, temp_gpt=0.0, max_new_tokens=1024, config=None):
    global key_manager

    if config:
        set_openai_config(config)

    if not key_manager:
        raise ValueError("API密钥管理器未初始化")

    try:
        key = key_manager.get_available_key()
        openai.api_key = key
        print(f"openai.api_key:{openai.api_key}")
        completion = openai.ChatCompletion.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temp_gpt,
            max_tokens=max_new_tokens,
        )
        reply = completion.choices[0].message.content
        total_tokens = completion.usage.total_tokens
        return reply, total_tokens
    except (requests.exceptions.Timeout, openai.error.Timeout) as e:
        print(f"请求超时,正在重试: {e}")
        raise
    except openai.error.AuthenticationError as e:
        print(f"API密钥认证失败: {e}")
        raise
    except openai.error.APIError as e:
        print(f"API错误: {e}")
        raise
    finally:
        if key_manager and "key" in locals():
            key_manager.release_key(key)


def build_comparison_prompt(summary: str, answer: str) -> str:
    prompt_template = """Compare these two responses to the same question:
Response 1 (to be verified): {summary}
Response 2 (correct answer): {answer}

Is Response 1 correct? Reply with only True/False/Unknown.
True = Response 1 is correct
False = Response 1 is incorrect
Unknown = Cannot determine

Answer with ONLY True/False/Unknown."""
    
    return prompt_template.format(summary=summary, answer=answer)


def process_json_file(input_path: str, output_path: str, args):
    processed_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for idx, item in enumerate(data, 1):
        if 'summary' not in item or 'answer' not in item:
            continue
            
        summary = item.get('summary', '')
        answer = item.get('answer', '')
        
        
        prompt = build_comparison_prompt(summary, answer)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        
        if response == 'true':
            item['new_correct'] = True
        elif response == 'false':
            item['new_correct'] = False
        else:
            item['new_correct'] = item['correct']
        
        log_message = f"Item {idx}:\n"
        log_message += f"Summary: {summary}\n"
        log_message += f"Answer: {answer}\n"
        log_message += f"Verify Response: {response}\n"
        log_message += f"New Correct: {item['new_correct']}\n"
        log_message += "-" * 50
        logging.info(log_message)
        
        processed_data.append(item)
        print(f"处理完成第 {idx} 条数据")
            
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def verify_correct(input_file, output_file):
    args = {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "temp_llm": 0.0,  
        "max_new_tokens": 1024,
        "config": {
            "openai_api_base": "https://api.deepinfra.com/v1/openai",
            "openai_keys": {
                # "KYgozM7pEA7KKhUvs9AJ0rRrPiuQbakp": 0, 
                "qHU7Lr4JOQWlIHt7vmqAH6UEI3eRW8La": 0,
                "SOfjAUp6gagi1dyRL2X3dcWPvduZjpMj": 0
            }
        }
    }
    
    logging.basicConfig(
        filename='LLaVA-REST-MCTS/logs/verification_process.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    process_json_file(
        input_path=input_file,
        output_path=output_file,
        args=args
    )

def format_choices(choices):
    formatted_choices = []
    for index, choice in enumerate(choices):
        # 将索引转换为字母
        letter = chr(65 + index)  # 65 是 'A' 的 ASCII 值
        formatted_choices.append(f"({letter}) {choice}")
    return ";\n".join(formatted_choices) + ";"

def process_lora_dataset(input_file, output_file):
    new_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 处理每个样本
    for sample in tqdm(data, desc="Processing Samples to lora data"):
        if not sample.get('correct', False):
            continue
        
        full_path = sample['image_path']
        image_path = full_path.split('vcr1images/')[-1] if 'vcr1images' in full_path else full_path
        question = sample['question']
        answer_choices = sample['answer_choices']
        options = format_choices(answer_choices)
        user_prompt = beam_sample_prompt.format(question=question, options=options)  
            
        new_sample = {
            "id": sample['data_id'],
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": user_prompt
                },
                {
                    "from": "gpt",
                    "value": sample['result']['value']
                }
            ]
        }
        
        new_data.append(new_sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
def process_vcr_result_policy_data(input_file, output_file, logger, only_correct=True):
    with jsonlines.open(input_file, 'r') as reader:
        for sample in tqdm(reader, desc="Processing Samples to lora data"):
            logger.info(f"Processing sample {sample['data_id']}")
            full_path = sample['image_path']
            image_path = full_path.split('vcr1images/')[-1] if 'vcr1images' in full_path else full_path
            question = sample['question']
            answer_choices = sample['answer_choices']
            options = format_choices(answer_choices)
            user_prompt = beam_sample_prompt.format(question=question, options=options)
            if len(sample['result']) == 0:
                continue
            for result in sample['result']:
                if only_correct and result['correct'] is False:
                    continue
                new_sample = {
                    "id": str(uuid.uuid4()),
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": user_prompt
                        },
                        {
                            "from": "gpt",
                            "value": result['value']
                        }
                    ]
                }
                
                with open(output_file, "a") as jsonl_file:
                    jsonl_file.write(json.dumps(new_sample) + "\n")
                break # 只选择1个postive样本
        logger.info(f"All samples are saved in {os.path.basename(output_file)}")
        
def process_scienceqa_result_policy_data(input_file, output_file, logger, only_correct=True):
    with jsonlines.open(input_file, 'r') as reader:
        for sample in tqdm(reader, desc="Processing Samples to lora data"):
            logger.info(f"Processing sample {sample['id']}")
            img_path = sample['img_path'] 
            if img_path:
                image_path = img_path.split('ScienceQA/')[-1] if 'ScienceQA' in img_path else img_path
            else:
                image_path = None
            if sample["hint"]: 
                question = sample['hint']+" "+sample['question']
            else:
                question = sample['question']
            answer_choices = sample['answer_choices']
            options = format_choices(answer_choices)
            if image_path:
                user_prompt = beam_sample_prompt.format(question=question, options=options)
                new_sample = {
                "id": str(uuid.uuid4()),
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": user_prompt
                    },
                    {
                        "from": "gpt",
                        "value": sample['format_solution']
                    }
                ]
            }
            else:
                user_prompt = beam_sample_prompt_no_image.format(question=question, options=options)
                new_sample = {
                    "id": str(uuid.uuid4()),
                    "conversations": [
                        {
                            "from": "human",
                            "value": user_prompt
                        },
                        {
                            "from": "gpt",
                            "value": sample['format_solution']
                        }
                    ]
                }
            
                
            with open(output_file, "a") as jsonl_file:
                jsonl_file.write(json.dumps(new_sample) + "\n")
        logger.info(f"All samples are saved in {os.path.basename(output_file)}")
    
def process_negative_result_sample_data(input_file, output_file, logger, only_incorrect=True):
    with jsonlines.open(input_file, 'r') as reader:
        for sample in tqdm(reader, desc="Processing Negative Samples to lora data"):
            logger.info(f"Processing sample {sample['data_id']}")
            full_path = sample['image_path']
            image_path = full_path.split('vcr1images/')[-1] if 'vcr1images' in full_path else full_path
            question = sample['question']
            answer_choices = sample['answer_choices']
            options = format_choices(answer_choices)
            user_prompt = beam_sample_prompt.format(question=question, options=options) 
            for result in sample['result']:
                if only_incorrect and not result['correct']:
                    new_sample = {
                        "data_id": str(uuid.uuid4()),
                        "image_path": image_path,
                        "question": user_prompt,
                        "answer": sample["answer_str"],
                        "solution": result["value"],
                        "correct": result["correct"],
                        "summary": None,
                    }
                    with open(output_file, "a") as jsonl_file:
                        jsonl_file.write(json.dumps(new_sample) + "\n")
        logger.info(f"All Negative samples are saved in {os.path.basename(output_file)}")
  
    
def process_positive_result_sample_data(input_file, output_file, logger):
    with jsonlines.open(input_file, 'r') as reader:
        for sample in tqdm(reader, desc="Processing Positive Samples to lora data"):
            logger.info(f"Processing sample {sample['id']}")
            id = sample['id']
            image_path = sample['image']
            question = sample['conversations'][0]['value']
            solution = sample['conversations'][1]['value']
            new_sample = {
                "data_id": id,
                "image_path": image_path,
                "question": question,
                "answer": None,
                "solution": solution,
                "correct": True,
                "summary": None,
            }
            with open(output_file, "a") as jsonl_file:
                jsonl_file.write(json.dumps(new_sample) + "\n")
        logger.info(f"All Positive samples are saved in {os.path.basename(output_file)}")
 
if __name__ == "__main__":
    # verify_correct()
    process_lora_dataset(input_file="LLaVA-REST-MCTS/output_verified.json",
                         output_file="LLaVA-REST-MCTS/output_lora_data.json")