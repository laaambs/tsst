import argparse
import yaml
from dataset.ve import VEDataset
from models.LLaVA.llava.train.train import train
from models.LLaVA.llava.train.train_value import train_value
from models.LLaVA.llava_inference import LLAVA, LlaVaProcessor
from self_train.eval import MLLM_evaluator
from self_train.eval_beam import BEAM_Json_Evaluator
from utils.assign_value import read_samples_from_json, assign_value_for_samples
from utils.verify_correct import verify_correct, process_lora_dataset, process_multiple_result_policy_data
from dataset.science_qa import ScienceQADataset
from dataset.vcr import VCRDataset
import os
import json
import re
import subprocess
import logging
import torch
import time
import sys
import random
import numpy as np

sys.path.append("./")

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse(data_config_path): 
    with open(data_config_path, "r", encoding="utf-8") as f:
        data_args = yaml.safe_load(f)
    return data_args

def setup_logging(log_file):
    logger = logging.getLogger(log_file)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def json_to_command(json_path, command):
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    for key, value in config.items():
        if isinstance(value, bool):
            value = str(value).capitalize()
        command.append(f"--{key}")
        command.append(str(value))
    
    return command
    
def process_command(command, logger):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    for line in process.stdout:
        print(line, end='')  
        logger.info(line.strip()) 
    
    stderr = process.stderr.read()
    if stderr:
        print("Error:", stderr)
        logger.error(stderr)

    process.wait()

def generate_policy_data(data_config_path):
    import models.model as model
    from run_MCTS import load_dataset, run_MCTS_by_dataset
    data_args = parse(data_config_path)
    output_file = data_args["output_file"]
    data_partition = data_args.get("data_partition")
    dataset_name = data_args.get("dataset_name")
    trace_file = os.path.join(os.path.dirname(output_file), f"{dataset_name}_{data_partition}.json")
    data_args["output_file"] = trace_file
    with open(data_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_args, f)
    if not os.path.exists(trace_file):
        dataset = load_dataset(data_args)
        run_MCTS_by_dataset(data_args, dataset, trace_file)
    verified_file = trace_file.replace("trace_data", "verified_data")
    if not os.path.exists(verified_file):
        verify_correct(trace_file, verified_file)
    policy_file = verified_file.replace("verified_data", "policy_data")
    if not os.path.exists(policy_file):
        process_lora_dataset(verified_file, policy_file)
        
    if model.llava_model is not None:
        print("del model.llava_model")
        del model.llava_model
        torch.cuda.empty_cache()

    if model.value_model is not None:
        print("del model.value_model")
        del model.value_model
        torch.cuda.empty_cache()

    return verified_file, policy_file


from models.LLaVA.llava.mm_utils import get_model_name_from_path
from models.LLaVA.llava.model.builder import load_pretrained_model

def generate_policy_data_beam(data_config_path, sample_config_path, policy_model_path, value_model_path):

    data_args = parse(data_config_path)
    sample_args = parse(sample_config_path)
    set_seed(int(data_args["seed"]))
    def dict_to_namespace(args_dict):
        return argparse.Namespace(**args_dict)
    sample_args = dict_to_namespace(sample_args)
    output_file = data_args["output_file"]
    data_partition = data_args.get("data_partition")
    dataset_name = data_args.get("dataset_name")

    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    trace_file = os.path.join(os.path.dirname(output_file), f"{dataset_name}_{data_partition}_{time_str}.jsonl")
    data_args["output_file"] = trace_file
    with open(data_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_args, f)
        
    logging_file = trace_file.replace("output", "logs").replace("trace_data", "generate_trace").replace(".jsonl", ".log")
    logger = setup_logging(logging_file)
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Sample arguments: {sample_args}")
    logger.info(f"Policy model: {os.path.basename(policy_model_path)}")
    logger.info(f"Value model: {os.path.basename(value_model_path)}")
    logger.info(f"Trace file name: {os.path.basename(trace_file)}")
    
    dataset_root = data_args['data_root']
    dataset_name = data_args['dataset_name']
    data_subset = data_args['data_subset']
    # data_partition = sample_args.data_partition
    # batch_size = 1
    sample_size = sample_args.sample_size # only for scienceqa
    seed = 42
    num_beams = sample_args.num_beams
    branch = sample_args.branch
    n_consistency = sample_args.n_consistency
    
    if 'vcr' in dataset_name:
        logger.info("Loading VCR dataset.")
        dataset = VCRDataset(dataset_root, dataset_name, data_subset, data_partition, caption_path=None)
        
    elif 'scienceqa' in dataset_name:
        logger.info("Loading ScienceQA dataset.")
        dataset = ScienceQADataset(dataset_root, dataset_name, sample_size=sample_size, seed=seed, data_partition=data_partition)
    elif 've' in dataset_name:
        logger.info("Loading VE dataset.")
        dataset = VEDataset(dataset_root, dataset_name, data_subset, data_partition, caption_path=None)
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported.")

    
    value_name = get_model_name_from_path(value_model_path)
    logger.info(f"Loading value model: {value_name}")
    
    _, value_model, _, _ = load_pretrained_model(
        model_path=value_model_path, model_base=None, model_name=value_name
    )
    
    logger.info("Initializing LLAVA model.")
    llava_model = LLAVA(policy_model_path)
    model, tokenizer, image_processor = llava_model.model, llava_model.tokenizer, llava_model.image_processor
    llava_processor = LlaVaProcessor(args=sample_args, model=model, tokenizer=tokenizer, image_processor=image_processor, mm_use_im_start_end=True)

    task = sample_args.task
    logger.info(f"Running beam search sampling for task: {task}")
    
    # run beam search sampling
    evaluator_for_sample = MLLM_evaluator(model, 
                                          tokenizer, 
                                          dataset, 
                                          value_model=value_model,
                                          batch_size=1, 
                                          device='cuda', 
                                          task=task, 
                                          processor=llava_processor, 
                                          image_processor=image_processor, 
                                          num_beams=num_beams, 
                                          branch=branch,
                                          n_consistency=n_consistency)

    res = evaluator_for_sample.beam_sample(trace_file, logger)
    logger.info("Beam sampling completed.")

    if llava_model is not None:
        print("del model.llava_model")
        del llava_model
        torch.cuda.empty_cache()

    if value_model is not None:
        print("del model.value_model")
        del value_model
        torch.cuda.empty_cache()

    return trace_file
    
def verify_correct(trace_file):
    verified_file = trace_file.replace("trace_data", "verified_data")
    
    trace_verifier = BEAM_Json_Evaluator(trace_file)
    
    logging_file = trace_file.replace("output", "logs").replace("trace_data", "verify_correct").replace(".jsonl", ".log")
    logger = setup_logging(logging_file)
    
    absolute_acc, voting_acc = trace_verifier.all_verify(logger, verified_file)
    
    return verified_file

def process_policy_data(verified_file):
    policy_file = verified_file.replace("verified_data", "policy_data")
    logging_file = policy_file.replace("output", "logs").replace("policy_data", "process_policy").replace(".jsonl", ".log")
    logger = setup_logging(logging_file)
    process_multiple_result_policy_data(verified_file, policy_file, logger, only_correct=True)
    return policy_file
    
def process_verified_all_trace_data(verified_file):
    verified_trace_file = verified_file.replace("verified_data", "verified_all_trace")
    logging_file = verified_trace_file.replace("output", "logs").replace("verified_all_trace", "process_policy").replace(".jsonl", ".log")
    logger = setup_logging(logging_file)
    process_multiple_result_policy_data(verified_file, verified_trace_file, logger, only_correct=False)
    return verified_trace_file
    
def finetune_policy(policy_config_path, policy_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # 根据生成的数据更新config文件
    with open(policy_config_path, 'r') as f:
        config = json.load(f)
    config['data_path'] = policy_file
    
    policy_data = os.path.splitext(os.path.basename(policy_file))[0]
    model_path = config['model_name_or_path']
    model_name = os.path.basename(model_path)
    match = re.search(r'.*\d+b', model_name)
    if match:
        model_name_base = match.group(0)
        new_model_name = f"{model_name_base}-{policy_data}"
        new_output_dir = os.path.join(os.path.dirname(model_path), new_model_name)
        config['output_dir'] = new_output_dir
        
    with open(policy_config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    command = ["deepspeed", "models/LLaVA/llava/train/train_mem.py"]
    command = json_to_command(policy_config_path, command)
    
    logging_file = policy_file.replace("output", "logs").replace("policy_data", "finetune_policy").replace(".json", ".log")
    logger = setup_logging(logging_file)
    
    process_command(command, logger)

    return new_output_dir
    
def generate_values(verified_all_trace):
    samples = read_samples_from_json(verified_all_trace)
    value_file = verified_all_trace.replace("verified_all_trace", "value_data")
    assign_value_for_samples(samples, value_file)
    return value_file

def finetune_value(value_config_path, value_file):
    with open(value_config_path, 'r') as f:
        config = json.load(f)
    config['data_path'] = value_file
    
    value_data = os.path.splitext(os.path.basename(value_file))[0]
    model_path = config['model_name_or_path']
    model_name = os.path.basename(model_path)
    match = re.search(r'.*prm', model_name)
    if match:
        model_name_base = match.group(0)
        new_model_name = f"{model_name_base}-{value_data}"
        new_output_dir = os.path.join(os.path.dirname(model_path), new_model_name)
        config['output_dir'] = new_output_dir
        
    with open(value_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    command = ["deepspeed", "models/LLaVA/llava/train/train_value_mem.py"]
    command = json_to_command(value_config_path, command)
    
    logging_file = value_file.replace("output", "logs").replace("value_data", "finetune_value").replace(".json", ".log")
    logger = setup_logging(logging_file)
    
    process_command(command, logger)

    return new_output_dir

def update_model_paths(policy_config_path, value_config_path, model_file_path):
    with open(policy_config_path, 'r') as f:
        policy_config = json.load(f)
    policy_model_path = policy_config.get('model_name_or_path', '')

    with open(value_config_path, 'r') as f:
        value_config = json.load(f)
    value_model_path = value_config.get('model_name_or_path', '')

    with open(model_file_path, 'r') as f:
        lines = f.readlines()

    with open(model_file_path, 'w') as f:
        for line in lines:
            if line.startswith("INFERENCE_MODEL_DIR"):
                f.write(f"INFERENCE_MODEL_DIR = '{policy_model_path}'\n")
            elif line.startswith("VALUE_BASE_MODEL_DIR"):
                f.write(f"VALUE_BASE_MODEL_DIR = '{value_model_path}'\n")
            else:
                f.write(line)

def train_policy_value():
    data_config_path = "config/data_config.yaml"
    policy_config_path = "config/policy_config.json"
    value_config_path = "config/value_config.json"
    model_file_path = "models/model.py"
    sample_config_path = "config/sample_config.yaml"
    policy_model_path = "llava-v1.5-7b-sft-policy-v2"
    value_model_path = "llava-v1.5-7b-sft-prm-v5-best"
    
    update_model_paths(policy_config_path, value_config_path, model_file_path)
    # verified_file, policy_file = generate_policy_data(data_config_path)
    trace_file = generate_policy_data_beam(data_config_path, sample_config_path, policy_model_path, value_model_path)
    verified_file, policy_file = verify_correct(trace_file)
    # policy_file = "output/policy_data/vcr_train_1000_1005.json"
    new_policy_adapter_path = finetune_policy(policy_config_path, policy_file)
    value_file = generate_values(verified_file)
    new_value_adapter_path = finetune_value(value_config_path, value_file)
    # TODO: 合并新policy及value，更新policy_config和value_config中的model_path
    return new_policy_adapter_path, new_value_adapter_path

def merge_json_files_to_jsonl(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    try:
                        for line in infile:
                            data = json.loads(line)
                            outfile.write(json.dumps(data) + '\n')
                        print(f"{file_path} is handled.")
                    except json.JSONDecodeError:
                        infile.seek(0) 
                        try:
                            data = json.load(infile)
                            # 如果是列表，逐个写入
                            if isinstance(data, list):
                                for item in data:
                                    outfile.write(json.dumps(item) + '\n')
                                print(f"{file_path} is handled.")
                            elif isinstance(data, dict):
                                outfile.write(json.dumps(data) + '\n')
                                print(f"{file_path} is handled.")
                            else:
                                print(f"文件 {filename} 的格式不支持。")
                        except json.JSONDecodeError as e:
                            print(f"文件 {filename} 解析错误: {e}")

def count_elements_in_jsonl(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError as e:
                print(f"解析错误: {e}")
    return count

def convert_jsonl_to_json(jsonl_file_path, json_file_path):
    data_list = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"解析错误: {e}")
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 生成仅正确的policy训练数据以及用于提取value的所有verified trace数据
    # policy_data = process_policy_data("output/verified_data/vcr_train_0_2999_20241210.jsonl")
    # element_count = count_elements_in_jsonl(policy_data)
    # print(f"policy_data文件中元素的个数: {element_count}")
    # convert_jsonl_to_json(policy_data, policy_data.replace('jsonl', 'json'))
    # verified_all_trace = process_verified_all_trace_data("output/verified_data/vcr_train_0_2999_20241210.jsonl")
    # element_count = count_elements_in_jsonl(verified_all_trace)
    # print(f"verified_all_trace文件中元素的个数: {element_count}")
    # convert_jsonl_to_json(verified_all_trace, verified_all_trace.replace('jsonl', 'json'))
    
    # policy_file = "output/policy_data/vcr_train_0_2999_20241210.json"
    # policy_config_path = "config/policy_config.json"
    # new_policy_adapter_path = finetune_policy(policy_config_path, policy_file)
    
    data_config_path = "config/data_config_ve.yaml"
    # policy_config_path = "config/policy_config.json"
    # value_config_path = "config/value_config.json"
    # model_file_path = "models/model.py"
    sample_config_path = "config/sample_config_ve.yaml"
    policy_model_path = "llava-v1.5-7b-sft-policy-v2"
    value_model_path = "llava-v1.5-7b-sft-prm-v5-best"
    # update_model_paths(policy_config_path, value_config_path, model_file_path)
    # # verified_file, policy_file = generate_policy_data(data_config_path)
    trace_file = generate_policy_data_beam(data_config_path, sample_config_path, policy_model_path, value_model_path)
    # verify_correct("output/trace_data/vcr_train_2140_2999_20241209003542.jsonl")
 
    # merge_json_files_to_jsonl(input_directory, output_file)
    
    # jsonl_file_path = 'output/policy_data/vcr_train_0_2999_20241210.jsonl'
    # json_file_path = 'output/policy_data/vcr_train_0_2999_20241210.json'
    # convert_jsonl_to_json(jsonl_file_path, json_file_path)
