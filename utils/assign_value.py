from __future__ import annotations  # 允许引用尚未定义的类
from dataclasses import dataclass
import json
import uuid
from tqdm import tqdm
# Assign value for the initial value dataset



# Assign value for generated samples
def assign_value_for_samples(samples, file_path):
    all_steps = []  
    cnt = 0
    

    for sample in samples:
        steps = sample_to_steps(sample)
        for s in steps:
            ss = s.get_llava_value_format()
            if ss['value'] == 0.0:
                cnt += 1
            all_steps.append(ss)  
    print(f"Zero value count: {cnt}")
    print(len(all_steps))


    with open(file_path, 'w') as f:
        json.dump(all_steps, f, indent=4)  
        
def read_samples_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    samples = []
    for element in tqdm(data, desc="Processing value samples", unit="element"):
        sample = line_to_sample(element)
        if sample:
            print(f"Processing {sample.data_id}")
            samples.append(sample)
    return samples


def line_to_sample(line_json):
    try:
        sample = Sample(**line_json, steps=get_steps_str_by_solution(line_json['solution']))
        return sample
    except Exception as e:
        print(f"Error creating Sample: {e}")
        return None
    
def sample_to_steps(sample):
    parent = None
    steps = []
    for i in range(len(sample.steps)):
        step = ValueStep(sample, i, parent=parent)
        steps.append(step)
        parent = step
    return steps
    
def get_steps_str_by_solution(solution):
    steps = solution.split('.')
    return [step + '.' for step in steps if step]  # 确保每个元素保留原本含有的句号






@dataclass
class Sample:
    solution: str
    summary: str
    correct: bool
    # new_correct: bool
    data_id: str
    image_path: str
    question: str
    answer: str
    steps: list


class ValueStep:
    v: float
    w: float
    parent: ValueStep | None
    sample: Sample
    
    def __init__(self, sample, step_index, parent):
        self.sample = sample
        self.step_index = step_index
        self.steps_so_far = sample.steps[:step_index+1]
        self.step = sample.steps[step_index]
        self.rsk = self.calculate_rsk()
        self.parent = parent
        if self.parent is None:
            self.v = float(0)
            self.w = float(0)
        else:
            self.w = float(self.calculate_wsk())
            self.v = float(self.calculate_vk())
        
    
    def get_dict(self):
        return {
            "step": self.step,
            "steps_so_far": ''.join(self.steps_so_far),
            "v": self.v,
            "w": self.w,
        }
    
    def get_llava_value_format(self):
        steps_so_far_str = ' '.join(self.steps_so_far)
        if self.sample.image_path is not None:
            return {
                "id": str(uuid.uuid4()),
                "image": self.sample.image_path,
                "conversations": [
                    {
                        'from': 'human',
                        'value': self.sample.question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'{steps_so_far_str}',
                    },
                ],
                'value': self.v,
            }
        else:
            return {
                "id": str(uuid.uuid4()),
                "conversations": [
                    {
                        'from': 'human',
                        'value': self.sample.question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'{steps_so_far_str}',
                    },
                ],
                'value': self.v,
            }
            
        
    def calculate_rsk(self):
        rsk = 0 if self.sample.correct else 1
        return rsk

    def calculate_wsk(self):
        assert self.parent is not None
        assert self.step_index != 0
        vk_minus_1 = self.parent.v
        mk = len(self.sample.steps) - self.step_index - 1
        return (1 - vk_minus_1) / (mk + 1) * (1 - 2 * self.rsk)


    def calculate_vk(self):
        if self.parent is None:
            assert self.step_index == 0
            return 0
        vk_minus_1 = self.parent.v
        return max(vk_minus_1 + self.w, 0)
            


if __name__ == "__main__":
    samples = read_samples_from_json("LLaVA-REST-MCTS/output/sample_data/vcr/vcr_train_0_2999_20241210.json")
    assign_value_for_samples(samples, "vcr_train_value_0_2999_20241210.json")
    print("Done!")