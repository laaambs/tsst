import json
import string
import re
import os
from tqdm import tqdm
from utils.verify_correct import call_gpt


class BEAM_Json_Evaluator:
    
    SCQA_VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if the provided response match the correct answer for the given question.
    - If provided response matches the correct answer in meaning and accuracy, reply 'True'
    - If provided response contradicts the correct answer, reply 'False'
    - If you cannot confidently determine the relationship, reply 'Unknown'
    Remember to ONLY reply with True/False/Unknown.
    """
    
    VERIFY_SYSTEM_PROMPT = """You are a precise evaluator. Your task is to determine if two responses convey the same meaning and correctness.
    - If Response 1 matches the correct answer in meaning and accuracy, reply 'True'
    - If Response 1 contradicts the correct answer, reply 'False'
    - If you cannot confidently determine the relationship, reply 'Unknown'
    Remember to ONLY reply with True/False/Unknown."""
    
    def __init__(self, file_path):
        self.file_path = file_path

        self.llm_verify_args = {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "temp_llm": 0.0,  
            "max_new_tokens": 1024,
            "config": {
                "openai_api_base": "https://api.deepinfra.com/v1/openai",
                "openai_keys": {
                    # "KYgozM7pEA7KKhUvs9AJ0rRrPiuQbakp": 0, 
                    # "qHU7Lr4JOQWlIHt7vmqAH6UEI3eRW8La": 0,
                    # "SOfjAUp6gagi1dyRL2X3dcWPvduZjpMj": 0,
                    "zlMZV3oaYXV0c9OqxTcj7pitUTFN0WFT": 0,
                    "jUPdi6SUwH3ftqzOv2yHWrbH9uL258vk": 0,
                    "qi0JfbGb5fQem8HN1nVOIrKxdZFo3iGT": 0,
                    "1zDZ5ljkWl5vXafOiuGzA09vS05Rh8BE": 0,
                    "lENEADLHgfLGvR2djiQLggmQXcXLhLvm": 0,
                    "bRn23RTpJ2suESY9wTGLAKNE0CYLZ6Jw": 0,
                    "8yCAAulwvcLI17IwGdPSfs02KCjk3sYq": 0,
                    "YGUYv4LuVB0MrGPToy06t43U7N9n8W6m": 0,
                    "oXeIyy6qVF3h7gIOSeWmf6qELaccjleM": 0,
                    "KGmS8fG3hw4VHox8hANhveI8bfWavzKm": 0,
                }
            }
        }
    
    def extract_highest_scores(self):
        file_path = self.file_path
        highest_scores = []

        # 判断文件类型（根据扩展名或内容）
        is_jsonl = os.path.splitext(file_path)[-1].lower() == ".jsonl"
        
        # 读取 JSONL 文件
        if is_jsonl:
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    # 找到得分最高的 result
                    best_result = max(data['result'], key=lambda x: x['score'])
                    highest_scores.append(best_result)
        else:  # 读取 JSON 文件
            with open(file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):  # 文件是列表的情况
                    for item in data:
                        best_result = max(item['result'], key=lambda x: x['score'])
                        highest_scores.append(best_result)
                else:  # 文件是单个字典的情况
                    best_result = max(data['result'], key=lambda x: x['score'])
                    highest_scores.append(best_result)
        
        return highest_scores
    
    def extract_highest_score_elements(self):
        file_path = self.file_path
        filtered_list = []

        # 判断文件类型（根据扩展名或内容）
        is_jsonl = os.path.splitext(file_path)[-1].lower() == ".jsonl"

        # 处理 JSONL 文件
        if is_jsonl:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    # 从 result 列表中找到最高分的元素
                    highest_score_result = max(data['result'], key=lambda x: x['score'])
                    # 创建一个新的字典，保留原始数据并替换 result 为最高分元素
                    filtered_data = {key: value for key, value in data.items() if key != 'result'}
                    filtered_data['result'] = highest_score_result
                    # 添加到过滤后的列表
                    filtered_list.append(filtered_data)
        else:  # 处理 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):  # 文件是列表的情况
                    for item in data:
                        highest_score_result = max(item['result'], key=lambda x: x['score'])
                        filtered_data = {key: value for key, value in item.items() if key != 'result'}
                        filtered_data['result'] = highest_score_result
                        filtered_list.append(filtered_data)
                else:  # 文件是单个字典的情况
                    highest_score_result = max(data['result'], key=lambda x: x['score'])
                    filtered_data = {key: value for key, value in data.items() if key != 'result'}
                    filtered_data['result'] = highest_score_result
                    filtered_list.append(filtered_data)

        return filtered_list
    
    def greedy_verify(self):
        # 提取得分最高的元素
        highesy_score_elements = self.extract_highest_score_elements()
        results = []
        
        # 使用 tqdm 添加进度条
        for element in tqdm(highesy_score_elements, desc="Verifying responses", unit="element"):
            response = element['result']['value']
            answer_str = element['answer_str']
            answer_label = element['answer_label']
            
            # 调用 verify 方法进行验证
            res = self.verify(response, answer_str, answer_label)
            element['correct'] = res
            print(f'verify res: {res}')
            
            results.append(res)
            acc = self.calculate_accuracy(results)
            print(f'accuracy: {acc}')

            
        
        # 计算准确率
        acc = self.calculate_accuracy(results)
        return results, acc, highesy_score_elements
       
    
    def extract_all_elements(self):
        file_path = self.file_path
        filtered_list = []

        # 判断文件类型（根据扩展名或内容）
        is_jsonl = os.path.splitext(file_path)[-1].lower() == ".jsonl"

        # 处理 JSONL 文件
        if is_jsonl:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    # # 从 result 列表中找到最高分的元素
                    # highest_score_result = max(data['result'], key=lambda x: x['score'])
                    # 创建一个新的字典，保留原始数据并替换 result 为最高分元素
                    filtered_data = {key: value for key, value in data.items()}
                    # filtered_data['result'] = highest_score_result
                    # 添加到过滤后的列表
                    filtered_list.append(filtered_data)
        else:  # 处理 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):  # 文件是列表的情况
                    for item in data:
                        # highest_score_result = max(item['result'], key=lambda x: x['score'])
                        filtered_data = {key: value for key, value in item.items()}
                        # filtered_data['result'] = highest_score_result
                        filtered_list.append(filtered_data)
                else:  # 文件是单个字典的情况
                    # highest_score_result = max(data['result'], key=lambda x: x['score'])
                    filtered_data = {key: value for key, value in data.items()}
                    # filtered_data['result'] = highest_score_result
                    filtered_list.append(filtered_data)

        return filtered_list
     
     
    def all_verify(self, logger, save_file):
        # 提取得分最高的元素
        all_elements = self.extract_all_elements()
        all_results = []
        
        absolute_correct = 0
        absolute_count = 0
        voting_correct = 0
        voting_count = 0
        greedy_correct = 0
        for element in tqdm(all_elements, desc="Verifying responses", unit="element"):
            logger.info(f"Verifying {element['data_id']}...")
            answer_str = element['answer_str']
            logger.info(f"answer: {answer_str}")
            results = []
            sorted_results = sorted(element['result'], key=lambda x: x['score'], reverse=True)
            for result in sorted_results:
                response = result['value']
                if response:
                    # 调用 verify 方法进行验证
                    res = self.verify_final_answer_by_llm(response, answer_str)
                else:
                    res = False
                result['correct'] = res
                logger.info(f'verify res: {res}')
                if res:
                    absolute_correct += 1
                results.append(res)
            if len(results)>0 and results[0]: # score最高的结果
                greedy_correct += 1
            absolute_count += len(results)
            all_results.append(results)
            voting_count += 1
            if sum(1 for r in results if r) >= (len(results) / 2):
                voting_correct += 1
            absolute_acc = round(absolute_correct/absolute_count,3)
            voting_acc = round(voting_correct/voting_count,3)
            greedy_acc = round(greedy_correct/voting_count, 3)
            logger.info(f'absolute accuracy: {absolute_acc}')
            logger.info(f'voting_acc: {voting_acc}')
            logger.info(f'greedy_acc: {greedy_acc}')
            with open(save_file, "a") as jsonl_file:
                jsonl_file.write(json.dumps(element) + "\n")
        return absolute_acc, voting_acc, greedy_acc
       
    
    def all_verify_beam(self, logger, save_file, dataset_name="vcr"):
        # 提取得分最高的元素
        all_elements = self.extract_all_elements()
        all_results = []
        
        absolute_correct = 0
        absolute_count = 0
        voting_correct = 0
        voting_count = 0
        greedy_correct = 0
        for element in tqdm(all_elements, desc="Verifying responses", unit="element"):
            logger.info(f"Verifying {element['data_id']}...")
            answer_str = element['answer_str']
            logger.info(f"answer: {answer_str}")
            results = []
            sorted_results = sorted(element['result'], key=lambda x: x['score'], reverse=True)
            for result in sorted_results:
                response = result['value']
                if dataset_name == "vcr":
                    res = self.verify(response, answer_str)
                elif dataset_name == "scienceqa":
                    res = self.scqa_verify(response=response, question=element["question"], 
                                        choices=element["answer_choices"],
                                        label=element["answer_label"])
            
                
                # res = self.verify(response, answer_str)
                # res = self.verify_final_answer_by_llm(response, answer_str)
                result['correct'] = res
                logger.info(f'verify res: {res}')
                if res:
                    absolute_correct += 1
                results.append(res)
            if len(results)>0 and results[0]: # score最高的结果
                greedy_correct += 1
            absolute_count += len(results)
            all_results.append(results)
            voting_count += 1
            if sum(1 for r in results if r) >= (len(results) / 2):
                voting_correct += 1
            absolute_acc = round(absolute_correct/absolute_count,3)
            voting_acc = round(voting_correct/voting_count,3)
            greedy_acc = round(greedy_correct/voting_count, 3)
            logger.info(f'absolute accuracy: {absolute_acc}')
            logger.info(f'voting_acc: {voting_acc}')
            logger.info(f'greedy_acc: {greedy_acc}')
            with open(save_file, "a") as jsonl_file:
                jsonl_file.write(json.dumps(element) + "\n")
        return absolute_acc, voting_acc, greedy_acc
       
    
    def verify(self, response, answer_str, answer_label=None):
        # res = self.verify_by_regular_expression(response, answer_label)
        # if res is None:
        res = self.verify_by_llm(response, answer_str)
        return res
    
    def generate_options_string(self, choices):
        # 获取字母序列 (A, B, C, ...)
        letters = string.ascii_uppercase
        # 检查是否足够选项字母
        if len(choices) > len(letters):
            raise ValueError("Too many choices for available letter options.")
        # 生成 "A. 选项内容" 的字符串列表
        options = [f"{(letters[i])}. {choice}" for i, choice in enumerate(choices)]
        # 将选项按行拼接为字符串
        return "\n".join(options).strip()
    
    def generate_label_string(self, label):
        # 获取字母序列 (A, B, C, ...)
        letters = string.ascii_uppercase
        # 检查是否足够选项字母
        if label > len(letters):
            raise ValueError("Label is too large for available letter options.")
        # 生成 "A. 选项内容" 的字符串列表
        return f"{(letters[label-1])}"
        
    
    def scqa_verify(self, response, question, choices, label):
        # label_letter = self.generate_label_string(label)
        # res = self.verify_by_regular_expression(response, label)
        # if res is None:
        choices_str = self.generate_options_string(choices)
        label_str = self.generate_label_string(label)
        answer_str = f"{label_str} {choices[label-1]}"
        res = self.scqa_verify_by_llm(response, question, choices_str, answer_str)
        return res
    
    def extract_model_answer_ABCDE(self, output_text):
        match = re.search(r"Final Answer:\s*(A|B|C|D|E)", output_text.strip())
        if match:
            return match.group(1)
        sentences = output_text.strip().split('.')
        last_sentence = sentences[-1].strip()
        match = re.search(r"\b(A|B|C|D|E)\b", last_sentence)
        if match:
            return match.group(1)
        return None
    
    def verify_by_regular_expression(self, response, answer_label):
        print('verify by regular expression')
        choice = self.extract_model_answer_ABCDE(response)
        if choice is None:
            return None
        choice_index = [i for i in string.ascii_uppercase].index(choice) + 1
        res = answer_label==choice_index
        print(f'verify res: {res}')
        return res
    
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


    def get_final_answer(self, response: str) -> str:
        match = re.search(r'final answer:\s*(.+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return self.extract_last_sentence(response)
    
    def verify_by_llm(self, response, answer_label: str):
        print('verify by llm')
        final_answer = self.get_final_answer(response)
        print(f'final_answer: {final_answer}')
        args=self.llm_verify_args
        print(f'answer_label str: {answer_label}')
        prompt = self.build_comparison_prompt(final_answer, answer_label)
        # print(f'prompt: {prompt}')
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
    
    
    def verify_final_answer_by_llm(self, response, answer_label: str):
        print('verify by llm')
        match = re.search(r'final answer:\s*(.+)', response, re.IGNORECASE)
        if match:
            final_answer = match.group(1).strip()
        else:
            return False #没有匹配到final answer，视为error
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
    
    
    def calculate_accuracy(self, results):
        correct = sum([1 if r else 0 for r in results])
        total = len(results)
        accuracy = correct / total
        return accuracy
    
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
    
    def build_scqa_comparison_prompt(self, response: str, question: str,
                                     choices: str, answer: str) -> str:
        prompt_template = """Here is a given multiple-choice question and its correct answer:\nQuestion: {question}\nOptions:\n{choices}\nCorrect Answer: {answer}\n\nHere is the response you need to evaluate:\nProvided Response: {response}\nIs Provided Response correct? Reply with only True/False/Unknown.
    True = Provided Response is correct
    False = Provided Response is incorrect
    Unknown = Cannot determine

    Answer with ONLY True/False/Unknown."""
        
        return prompt_template.format(response=response, question=question, choices=choices, answer=answer)
    