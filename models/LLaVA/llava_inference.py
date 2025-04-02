from ast import List
from typing import Union, List
from llava.model.builder import load_pretrained_model

from llava.mm_utils import get_model_name_from_path
import torch
from llava.conversation import (SeparatorStyle,
                                        conv_templates)
from llava.mm_utils import KeywordsStoppingCriteria
from transformers import StoppingCriteria

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


class LLAVA:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_name
        )
    
    
    def infer(self, query, image_file, conv_mode=None, temperature=0.1, top_p=None, num_beams=1, max_new_tokens=1024):
        # Model
        disable_torch_init()

        qs = query
        
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        if image_file:
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, "", qs)

        if "llama-2" in self.model_name.lower():
            _conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            _conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            _conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            _conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            _conv_mode = "mpt"
        else:
            _conv_mode = "llava_v0"

        if conv_mode is not None and _conv_mode != conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    _conv_mode, conv_mode, conv_mode
                )
            )
        else:
            conv_mode = _conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if not image_file:
            images_tensor = None
            image_sizes = None
        else:
            image_files = self.image_parser(image_file)
            images = self.load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                self.image_processor,
                self.model.config
            ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(outputs)
        return outputs
        
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


    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out
    



class LlaVaProcessor:
    def __init__(self, args, model, tokenizer, image_processor, mm_use_im_start_end):
        self.mm_use_im_start_end = mm_use_im_start_end
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = "llava_v1"
        self.args = args

    def load_demo_images(image_files: Union[List[str], str]):
        if type(image_files) is list:
            out = []
            for image_file in image_files:
                image = Image.open(image_file).convert("RGB")
                out.append(image)
        else:
            out = Image.open(image_files).convert("RGB")
        return out

    def format_text(self, text: str):
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text
    

    
    def prepare_prompt_image_token_beam_sample(self, prompt):
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
        if self.model.config.mm_use_im_start_end:
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, image_token_se)
        
        return prompt
    
    def prepare_prompt_beam_sampling(self, query, solution:str):
        query = self.prepare_prompt_image_token_beam_sample(query)
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], solution)
        prompt = conv.get_prompt()
        
        return prompt

    
    

    def load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

    def get_processed_tokens(self, text: str, image_path: str):
        prompt = self.format_text(text)
        image = self.load_image(image_path)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return image_tensor, input_ids

    def get_processed_tokens_batch(self, batch_text: List[str], image_paths: List[str]):
        
        prompt = [self.format_text(text) for text in batch_text]
        
        images = [
            self.load_image(image_path) if image_path is not None else torch.zeros((3, 336, 336))
            for image_path in image_paths
        ]
        
        image_tensors = []
        for img in images:
            if isinstance(img, torch.Tensor):  # 如果是空张量，直接加入
                image_tensors.append(img)
            else:  # 否则进行正常的图像预处理
                image_tensor = self.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                image_tensors.append(image_tensor)
                # 将图像张量堆叠成批量
        batch_image_tensor = torch.stack(image_tensors)

        # 处理文本，生成输入 ID
        batch_input_ids = [
            tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for p in prompt
        ]

        # 对输入 ID 进行 Padding
        max_len = max(len(seq) for seq in batch_input_ids)
        padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids)

        return batch_image_tensor, batch_input_ids    
        
            

    def batch_inference(self, batch):
        conv = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)]
            if conv.version == "v0"
            else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        input_ids = input_ids.cuda()
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_length,
                # length_penalty=self.args.length_penalty,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                do_sample=self.args.do_sample,
                temperature=self.args.temperature,
                num_return_sequences=self.args.num_return_sequences,
            )
            
        generated_outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        # generated_outputs = self.tokenizer.batch_decode(
        #     output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        # )
        generated_outputs = [out.strip() for out in generated_outputs]
        generated_outputs = [
            out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs
        ]
        return generated_outputs
    
    def batch_inference_beam(self, batch, num_beams):
               
        input_ids = batch["input_ids"]
        input_ids = input_ids.cuda()
        image_tensor = batch["image_tensors"].to('cuda', dtype=torch.float16)
        
        batch_size=input_ids.shape[0]
        stopping_criteria = self.prepare_stopping_criteria( batch_size,num_beams)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                images=image_tensor,
                # image_sizes=image_sizes,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_length,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                early_stopping=True, 
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        

        candidate_sequences = stopping_criteria.stopped_sequences_sets
        
        return candidate_sequences, output
    
    
    def prepare_stopping_criteria(self, batch_size, num_beams, stop_words_ids=torch.tensor([[29889]])):
        """准备停止条件"""
        stop_words_ids = stop_words_ids.to('cuda')
        return BEAMKeywordsStoppingCriteria(
            tokenizer=self.tokenizer,
            stops=stop_words_ids,
            batch_size=batch_size, # pass in args
            num_beams=num_beams
        )


class BEAMKeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops = [], batch_size = 1, num_beams = 1):
        super().__init__() 
        self.tokenizer = tokenizer
        self.stops = stops
        self.stopped_sequences_sets = [set() for _ in range(batch_size)]
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.stop_flags = torch.zeros(batch_size*num_beams, dtype=torch.bool, device=stops.device)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.stop_flags.all():
            return True
        
        n = input_ids.shape[0]
        
        
        for batch_idx in range(self.batch_size):
            
            
            for beam_idx in range(self.num_beams):
                if self.stop_flags[batch_idx*self.num_beams + beam_idx]:
                    continue
                
                current_ids = input_ids[batch_idx*self.num_beams + beam_idx]
                for stop in self.stops:
                    if stop.shape[0] == 1:
                        matches = torch.where(current_ids == stop.item())[0]
                        if len(matches) > 0:
                            stop_index = matches[0].item() + 1
                            self._process_stop(current_ids, stop_index, batch_idx, beam_idx)
                            break
                    else:
                        for i in range(len(current_ids) - len(stop) + 1):
                            if torch.equal(current_ids[i:i+len(stop)], stop):
                                self._process_stop(current_ids, i+len(stop), batch_idx, beam_idx)
                                break
        
        # for batch_idx in range(batch_size):
        #     if self.stop_flags[batch_idx]:
        #         continue
            
        #     current_ids = input_ids[batch_idx]
        #     for stop in self.stops:
        #         if stop.shape[0] == 1:
        #             matches = torch.where(current_ids == stop.item())[0]
        #             if len(matches) > 0:
        #                 stop_index = matches[0].item() + 1
        #                 self._process_stop(current_ids, stop_index, batch_idx)
        #                 break
        #         else:
        #             for i in range(len(current_ids) - len(stop) + 1):
        #                 if torch.equal(current_ids[i:i+len(stop)], stop):
        #                     self._process_stop(current_ids, i+len(stop), batch_idx)
        #                     break
        return bool(self.stop_flags.all().item()) 
    
    def _process_stop(self, input_ids, stop_index, batch_idx, beam_idx):
        if stop_index == 0:
            decoded_output = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            decoded_output = self.tokenizer.decode(input_ids[:stop_index], skip_special_tokens=True)
        self.stopped_sequences_sets[batch_idx].add(decoded_output.strip())
        self.stop_flags[batch_idx*beam_idx+beam_idx] = True 
   


if __name__ == "__main__":
        
    # model_path = "liuhaotian/llava-v1.6-vicuna-13b"
    model_path = "/home/wt/.cache/huggingface/hub/models--zhiqings--LLaVA-RLHF-7b-v1.5-224/snapshots/e209b5158897700108cefae829393079dbea0416/sft_llava_rlhf_model"
    image_url = "./layout/examples/basic.jpg"
    prompt = """
    This is a building layout planning diagram. The purple circles represent possible positions for placing tower cranes, each labeled with a unique number. Step-by-step, analyze which purple circle would provide the highest transportation efficiency if a tower crane were placed there.

    Output format: Step 1, Step 2, etc., ending with the final selected number.
    """
    
    # prompt = 'describe this image'
    
    llava_inference = LLAVA(model_path)
    output = llava_inference.infer(prompt, image_url)
    print(output)
    
    # python -m models.llava
