from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch

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
    
    
    def value(self, query, image_file, conv_mode=None, temperature=0, top_p=None, num_beams=1, max_new_tokens=1024):
        # Model
        disable_torch_init()

        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
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
            outputs = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                # do_sample=True if temperature > 0 else False,
                # temperature=temperature,
                # top_p=top_p,
                # num_beams=num_beams,
                # max_new_tokens=max_new_tokens,
                # use_cache=True,
            )

        # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
