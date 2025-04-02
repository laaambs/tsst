import torch
import requests
from PIL import Image
from io import BytesIO
import re
from transformers import StoppingCriteria
# from models.LLaVA.llava_inference import LLAVA, LlaVaProcessor
import random
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


# class KeywordsStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, stops = [], batch_size = 1, num_beams = 1):
#         super().__init__() 
#         self.tokenizer = tokenizer
#         self.stops = stops
#         self.batch_size = batch_size
#         self.num_beams = num_beams
#         self.stopped_sequences = [set() for _ in range(batch_size)]
#         self.stop_flags = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=stops.device)

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         if self.stop_flags.all():
#             return True
        
#         sample_num = input_ids.shape[0]
        
#         for idx in range(sample_num):
#             if self.stop_flags[idx]:
#                 continue
            
#             current_ids = input_ids[idx]
#             for stop in self.stops:
#                 if stop.shape[0] == 1:
#                     matches = torch.where(current_ids == stop.item())[0]
#                     if len(matches) > 0:
#                         stop_index = matches[0].item() + 1
#                         self._process_stop(current_ids, stop_index, idx)
#                         break
#                 else:
#                     for i in range(len(current_ids) - len(stop) + 1):
#                         if torch.equal(current_ids[i:i+len(stop)], stop):
#                             self._process_stop(current_ids, i+len(stop), idx)
#                             break
#         return bool(self.stop_flags.all().item()) 
    
#     def _process_stop(self, input_ids, stop_index, idx):
#         if stop_index == 0:
#             decoded_output = self.tokenizer.decode(input_ids, skip_special_tokens=True)
#         else:
#             decoded_output = self.tokenizer.decode(input_ids[:stop_index], skip_special_tokens=True)
#         self.stopped_sequences[idx // self.num_beams].add(decoded_output.strip())
#         self.stop_flags[idx] = True 
 
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops = [], batch_size = 1):
        super().__init__() 
        self.tokenizer = tokenizer
        self.stops = stops
        self.stopped_sequences = set()
        self.stop_flags = torch.zeros(batch_size, dtype=torch.bool, device=stops.device)

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
    
    def _process_eos_stop(self, input_ids, stop_index, batch_idx):
        if stop_index == 0:
            decoded_output = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            decoded_output = self.tokenizer.decode(input_ids[:stop_index], skip_special_tokens=True)
        self.stopped_sequences.add(decoded_output.strip()+"</s>")
        self.stop_flags[batch_idx] = True 
    
def check_end_condition(text):
    end_tokens = [
        "</s>", "###", "---", "final answer", "user", "human", ". . ."
    ]
    max_length = 1000
    end_pattern = re.compile("|".join(map(re.escape, end_tokens)), re.IGNORECASE)
    return bool(end_pattern.search(text)) or len(text) > max_length

def prepare_prompt_image_token(model, prompt):
    if DEFAULT_IMAGE_TOKEN in prompt:
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        prompt = prompt.strip()
    elif IMAGE_PLACEHOLDER in prompt:
        prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
    else:
        prompt = prompt.strip()
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        prompt = prompt.strip()
    if model.config.mm_use_im_start_end:
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, image_token_se)
    
    return prompt

def set_conv_mode(model_name, args):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    return args

def prepare_prompt(args, model, query, solution=None):
    query = prepare_prompt_image_token(model, query)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], solution)
    prompt = conv.get_prompt()
    
    return prompt

# TODO: check pre-padding or post-padding?
def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype, device=sequence.device), sequence])

def eval_model(args, model, tokenizer, image_processor, model_name, value_model, value_name, logger):
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    args = set_conv_mode(model_name, args)
    
    query = args.query
    candidate_solutions = [{"value":None,"score":0.0}]
    final_solutions = []
    
    stop_words_ids=torch.tensor([[2], [29889]])
    # stop_words = ["."]
    count = 0
    while(len(final_solutions)<args.n_consistency):
        extended_candidate_solutions = []

        # batch_input_ids = []
        # for solution in candidate_solutions:
        #     prompt = prepare_prompt(args=args, model=model, 
        #                         query=query, solution=solution["value"])
        #     input_ids = (
        #             tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        #             .unsqueeze(0)
        #             .cuda()
        #         )

        #     batch_input_ids.append(input_ids.squeeze())
            
        # batch_size = len(candidate_solutions)
        # batch_image_tensor = torch.cat([images_tensor] * batch_size, dim=0)
        # batch_image_sizes = image_sizes * batch_size
            
        # max_len = max(len(seq) for seq in batch_input_ids)
        # padded_input_ids = [pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        # batch_input_ids = torch.stack(padded_input_ids)
        
        # stopping_criteria = KeywordsStoppingCriteria(tokenizer=tokenizer, stops=stop_words_ids, batch_size=batch_size, num_beams=args.num_beams)

        # # TODO: policy的推理改成 batch inference
        # with torch.inference_mode():
        #     output = model.generate(
        #         batch_input_ids,
        #         images=batch_image_tensor,
        #         image_sizes=batch_image_sizes,
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         max_new_tokens=args.max_new_tokens,
        #         use_cache=True,
        #         stopping_criteria=[stopping_criteria],
        #         early_stopping=True, 
        #         num_beams=args.num_beams,
        #         num_return_sequences=args.num_beams,
        #         output_scores=True,
        #         return_dict_in_generate=True,
        #     )
        
        # candidate_sequences = stopping_criteria.stopped_sequences
    
        # for solution, sequences in zip(candidate_solutions, candidate_sequences):
        #     if solution["value"] is None:
        #         extended_candidate_solutions.extend(sequences)
        #     else:
        #         extended_candidate_solutions.extend(solution["value"] + " " + seq for seq in sequences)
        
        
        for solution in candidate_solutions:
            prompt = prepare_prompt(args=args, model=model, 
                                query=query, solution=solution["value"])
    
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )
        
            stopping_criteria = KeywordsStoppingCriteria(tokenizer=tokenizer, stops=stop_words_ids, batch_size=input_ids.shape[0] * args.num_beams)

            # TODO: policy的推理改成 batch inference
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    early_stopping=True, 
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        
            candidate_sequences = stopping_criteria.stopped_sequences
    
            if solution["value"] is None:
                for candidate_sequence in candidate_sequences:
                    extended_candidate_solutions.append(candidate_sequence)
            else:
                for candidate_sequence in candidate_sequences:
                    extended_candidate_solutions.append(solution["value"] + " " + candidate_sequence)
        
        candidate_scores = [0] * len(extended_candidate_solutions)
        
        # TODO: value的推理改为batch inference
        for i, extended_candidate_solution in enumerate(extended_candidate_solutions):
            value_prompt = prepare_prompt(args, model, query, extended_candidate_solution)
            value_input_ids = (
                tokenizer_image_token(value_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                candidate_score = value_model.generate(
                    value_input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                )
            candidate_scores[i] = candidate_score.item()
            
        
        # batch_input_ids = []
        # for i, extended_candidate_solution in enumerate(extended_candidate_solutions):
        #     value_prompt = prepare_prompt(args, model, query, extended_candidate_solution)
        #     value_input_ids = (
        #         tokenizer_image_token(value_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        #         .unsqueeze(0)
        #         .cuda()
        #     )
        #     batch_input_ids.append(value_input_ids.squeeze())
            
        # batch_size = len(batch_input_ids)
        # batch_image_tensor = torch.cat([images_tensor] * batch_size, dim=0)
        # batch_image_sizes = image_sizes * batch_size
        
        # max_len = max(len(seq) for seq in batch_input_ids)
        # padded_input_ids = [pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        # batch_input_ids = torch.stack(padded_input_ids)
        
        # with torch.inference_mode():
        #     candidate_scores = value_model.generate(
        #         batch_input_ids,
        #         images=batch_image_tensor,
        #         image_sizes=batch_image_sizes,
        #     )
            
        # candidate_scores = candidate_scores.tolist()
        
        candidate_solutions = []
        for value, score in zip(extended_candidate_solutions, candidate_scores):
            candidate_solutions.append({
                'value': value,
                'score': score
            })
        
        candidate_solutions = sorted(candidate_solutions, key=lambda x: x['score'], reverse=True)
        
        print(f"step {count}, candidate_solutions: {candidate_solutions}")
        
        if len(candidate_solutions)>args.branch:
            candidate_solutions = candidate_solutions[:args.branch]
        
        for solution in candidate_solutions[:]:
            if check_end_condition(solution['value']):
                final_solutions.append(solution)
                candidate_solutions.remove(solution)
                
        print(f"step {count}, final_solutions: {final_solutions}")
        count+=1
        
        if len(candidate_solutions)==0:
            candidate_solutions = [{"value":None,"score":0.0}]
            
            
        
        # candidate_solutions = [solution + candidate_sequence for candidate_sequence in candidate_sequences]
        # value_prompts = [prepare_prompt(args, model, query, sol) for sol in candidate_solutions]
        # value_input_ids = torch.cat([
        #     tokenizer_image_token(value_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        #     for value_prompt in value_prompts
        # ], dim=0).cuda()
        
        # batch_size = len(candidate_solutions)
        # images_tensor_batch = [images_tensor for _ in range(batch_size)]
        # image_sizes_batch = image_sizes * batch_size
        # with torch.inference_mode():
        #     candidate_scores = value_model.generate(
        #         value_input_ids,
        #         images=images_tensor_batch,
        #         image_sizes=image_sizes_batch,
        #     )
        #     candidate_scores = [score.item() for score in candidate_scores]

    return sorted(final_solutions, key=lambda x: x['score'], reverse=True)
   



def eval_model_copy(args, model, tokenizer, image_processor, model_name, value_model, value_name, logger):
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    args = set_conv_mode(model_name, args)
    
    query = args.query
    candidate_solutions = [{"value":None,"score":0.0}]
    final_solutions = []
    
    stop_words_ids=torch.tensor([[29889]])
    # stop_words = ["."]
    count = 0
    while(len(final_solutions)<args.n_consistency):
        extended_candidate_solutions = []
        
        for solution in candidate_solutions:
            prompt = prepare_prompt(args=args, model=model, 
                                query=query, solution=solution["value"])
    
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )
        
            stopping_criteria = KeywordsStoppingCriteria(tokenizer=tokenizer, stops=stop_words_ids, batch_size=input_ids.shape[0] * args.num_beams)

            # TODO: policy的推理改成 batch inference
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    early_stopping=True, 
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        
            candidate_sequences = stopping_criteria.stopped_sequences
    
            if solution["value"] is None:
                for candidate_sequence in candidate_sequences:
                    extended_candidate_solutions.append(candidate_sequence)
            else:
                for candidate_sequence in candidate_sequences:
                    extended_candidate_solutions.append(solution["value"] + " " + candidate_sequence)
        
        candidate_scores = [0] * len(extended_candidate_solutions)
        
        # TODO: value的推理改为batch inference
        for i, extended_candidate_solution in enumerate(extended_candidate_solutions):
            value_prompt = prepare_prompt(args, model, query, extended_candidate_solution)
            value_input_ids = (
                tokenizer_image_token(value_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                candidate_score = value_model.generate(
                    value_input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                )
            candidate_scores[i] = candidate_score.item()

        candidate_solutions = []
        for value, score in zip(extended_candidate_solutions, candidate_scores):
            candidate_solutions.append({
                'value': value,
                'score': score
            })
        
        candidate_solutions = sorted(candidate_solutions, key=lambda x: x['score'], reverse=True)
        
        print(f"step {count}, candidate_solutions: {candidate_solutions}")
        
        if len(candidate_solutions)>args.branch:
            candidate_solutions = candidate_solutions[:args.branch]
        
        for solution in candidate_solutions[:]:
            if check_end_condition(solution['value']):
                final_solutions.append(solution)
                candidate_solutions.remove(solution)
                
        print(f"step {count}, final_solutions: {final_solutions}")
        count+=1
        
        if len(candidate_solutions)==0:
            candidate_solutions = [{"value":None,"score":0.0}]

    return sorted(final_solutions, key=lambda x: x['score'], reverse=True)
   


def topk_beam_sample(batch, policy_model, tokenizer, image_processor, value_model, batch_size):
    # TODO
    image_files = []
    
    
    
    pass


def beam_sample(args, model, tokenizer, image_processor, model_name, value_model, batch_size, value_name, logger):
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    args = set_conv_mode(model_name, args)
    
    query = args.query
    candidate_solutions = [{"value":None,"score":0.0}]
    final_solutions = []
    
    stop_words_ids=torch.tensor([[29889]])
    # stop_words = ["."]
    count = 0
    while(len(final_solutions)<args.n_consistency):
        extended_candidate_solutions = []
        for solution in candidate_solutions:
            prompt = prepare_prompt(args=args, model=model, 
                                query=query, solution=solution["value"])
    
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )
        
            stopping_criteria = KeywordsStoppingCriteria(tokenizer=tokenizer, stops=stop_words_ids, batch_size=input_ids.shape[0] * args.num_beams)

            # TODO: policy的推理改成 batch inference
            # LLava库里有没有现有的batch inference代码?
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    early_stopping=True, 
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        
            candidate_sequences = stopping_criteria.stopped_sequences
    
            if solution["value"] is None:
                for candidate_sequence in candidate_sequences:
                    extended_candidate_solutions.append(candidate_sequence)
            else:
                for candidate_sequence in candidate_sequences:
                    extended_candidate_solutions.append(solution["value"] + " " + candidate_sequence)
        
        candidate_scores = [0] * len(extended_candidate_solutions)
        
        # TODO: value的推理改为batch inference
        for i, extended_candidate_solution in enumerate(extended_candidate_solutions):
            value_prompt = prepare_prompt(args, model, query, extended_candidate_solution)
            value_input_ids = (
                tokenizer_image_token(value_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                candidate_score = value_model.generate(
                    value_input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                )
            candidate_scores[i] = candidate_score.item()
        
        candidate_solutions = []
        for value, score in zip(extended_candidate_solutions, candidate_scores):
            candidate_solutions.append({
                'value': value,
                'score': score
            })
        
        candidate_solutions = sorted(candidate_solutions, key=lambda x: x['score'], reverse=True)
        
        print(f"step {count}, candidate_solutions: {candidate_solutions}")
        
        if len(candidate_solutions)>args.branch:
            candidate_solutions = candidate_solutions[:args.branch]
            # scores = [solution['score'] for solution in candidate_solutions]
            # total_score = sum(scores)
            # probabilities = [score / total_score for score in scores]
            # selected_indices = random.choices(range(len(candidate_solutions)), weights=probabilities, k=args.branch)
            # candidate_solutions = [candidate_solutions[i] for i in selected_indices]
        
        for solution in candidate_solutions[:]:
            if check_end_condition(solution['value']):
                final_solutions.append(solution)
                candidate_solutions.remove(solution)
                
        print(f"step {count}, final_solutions: {final_solutions}")
        count+=1
        
        if len(candidate_solutions)==0:
            candidate_solutions = [{"value":None,"score":0.0}]
            

    return sorted(final_solutions, key=lambda x: x['score'], reverse=True)
   
   
   

