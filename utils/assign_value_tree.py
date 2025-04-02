import json
import re
import string
from typing import List, Optional
import math
import uuid
import argparse

class TreeNode:
    """
    A class representing a node in a tree.
    """
    def __init__(self, value: str, step: str = "", correct: bool = False):
        """
        Initialize a tree node with a value, step information, and correctness flag.
        
        :param value: The value of the tree node
        :param step: A string representing the step information
        :param correct: Boolean indicating if this node is a correct leaf
        """
        self.value: str = value
        self.children: List['TreeNode'] = []
        self.step: str = step
        self.correct: bool = correct
        self.mk: float = math.inf
        self.correct_reachable: bool = False
        self.rsk: float = 1.0  # Default to 1, updated later
        self.wsk: float = 0.0
        self.vk: float = 0.0
        self.parent: Optional['TreeNode'] = None  # To keep track of parent node

    def add_child(self, child_node: 'TreeNode'):
        """
        Add a child node to this node and set its parent.
        
        :param child_node: An instance of TreeNode to be added as a child
        """
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node: 'TreeNode'):
        """
        Remove a child node from this node.
        
        :param child_node: An instance of TreeNode to be removed
        """
        self.children = [child for child in self.children if child != child_node]

    def traverse(self) -> List[str]:
        """
        Perform a depth-first traversal of the tree starting from this node.
        
        :return: A list of values in depth-first order
        """
        nodes = []
        stack = [self]
        while stack:
            current = stack.pop()
            nodes.append(current.value)
            stack.extend(reversed(current.children))
        return nodes
    
    def format_node(self, root, img_path) -> dict:
        """
        Format the node's data into the specified JSON-like structure.
        
        :return: A dictionary representing the formatted node data
        """
        return {
            "id": str(uuid.uuid4()),
            "image": img_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n"+root.value
                },
                {
                    "from": "gpt",
                    "value": self.value
                }
            ],
            "value": self.vk
        }

class Tree:
    """
    A class representing a tree structure.
    """
    def __init__(self, root_value: str, img_path: Optional[str] = None):
        """
        Initialize a tree with a root node.
        
        :param root_value: The value of the root node
        """
        self.root = TreeNode(root_value)
        self.img_path = img_path

    def update_tree(self):
        """
        Update the entire tree's mk, correct_reachable, rsk, wsk, and vk values based on the specified algorithm.
        """
        # Step 1: Post-order traversal to update mk and correct_reachable
        self._post_order_update(self.root)
        
        # Step 1.2: Set mk for nodes with mk=inf to the nearest ancestor's mk
        self._set_inf_mk(self.root, inherited_mk=self.root.mk)
        
        # Check if all mk are inf
        all_inf = self._check_all_inf(self.root)
        if all_inf:
            # Set all vk to 0
            self._set_vk_zero(self.root)
            return
        
        # Step 3: Traverse from root to compute wsk and vk
        self._compute_vk(self.root, parent_vk=0.0)

    def _post_order_update(self, node: TreeNode):
        """
        Perform a post-order traversal to update mk and correct_reachable for each node.
        
        :param node: The current TreeNode
        """
        if not node.children:
            # Leaf node
            if node.correct:
                node.mk = 0
                node.correct_reachable = True
            else:
                node.mk = math.inf
                node.correct_reachable = False
        else:
            min_mk = math.inf
            any_reachable = False
            for child in node.children:
                self._post_order_update(child)
                if child.correct_reachable:
                    any_reachable = True
                    if child.mk + 1 < min_mk:
                        min_mk = child.mk + 1
            node.correct_reachable = any_reachable
            node.mk = min_mk if any_reachable else math.inf

    def _set_inf_mk(self, node: TreeNode, inherited_mk: float):
        """
        Traverse the tree and set mk to the nearest ancestor's mk if mk is inf.
        
        :param node: The current TreeNode
        :param inherited_mk: The inherited mk value from the nearest ancestor
        """
        if node.mk != math.inf:
            # Current node has a finite mk, update inherited_mk
            current_inherited_mk = node.mk
        else:
            # Current node has mk=inf, set it to inherited_mk
            node.mk = inherited_mk
            current_inherited_mk = inherited_mk

        for child in node.children:
            self._set_inf_mk(child, current_inherited_mk)

    def _check_all_inf(self, node: TreeNode) -> bool:
        """
        Check if all mk in the tree are inf.
        
        :param node: The current TreeNode
        :return: True if all mk are inf, else False
        """
        if node.mk != math.inf:
            return False
        for child in node.children:
            if not self._check_all_inf(child):
                return False
        return True

    def _set_vk_zero(self, node: TreeNode):
        """
        Set vk to 0 for all nodes in the tree.
        
        :param node: The current TreeNode
        """
        node.vk = 0.0
        node.wsk = 0.0
        for child in node.children:
            self._set_vk_zero(child)

    def _compute_vk(self, node: TreeNode, parent_vk: float):
        """
        Traverse the tree from root to compute wsk and vk for each node.
        
        :param node: The current TreeNode
        :param parent_vk: The vk value of the parent node
        """
        # Compute rsk
        node.rsk = 0.0 if node.correct_reachable else 1.0
        
        # Compute wsk
        node.wsk = (1 - parent_vk) * (1 - 2 * node.rsk) / (node.mk + 1)
        
        # Compute vk
        node.vk = 0.0 if node.parent is None else max(parent_vk + node.wsk, 0.0)
        
        # Traverse children
        for child in node.children:
            self._compute_vk(child, node.vk)

    def print_all_nodes(self):
        """
        Print all nodes in the tree with their attributes.
        """
        nodes = []
        self._traverse_and_collect(self.root, nodes)
        
        for node in nodes:
            print(f"Value: {node.value}")
            print(f"  Step: {node.step}")
            print(f"  Correct: {node.correct}")
            print(f"  mk: {node.mk}")
            print(f"  correct_reachable: {node.correct_reachable}")
            print(f"  rsk: {node.rsk}")
            print(f"  wsk: {node.wsk}")
            print(f"  vk: {node.vk}")
            print("-" * 40)
            
        print(f'len(nodes) {len(nodes)}')

    def _traverse_and_collect(self, node: TreeNode, nodes: List[TreeNode]):
        """
        Collect all nodes in the tree using depth-first traversal.
        
        :param node: The current TreeNode
        :param nodes: List to collect nodes
        """
        nodes.append(node)
        for child in node.children:
            self._traverse_and_collect(child, nodes)
            
    def _traverse_and_collect_formatted(self, root, node: TreeNode, img_path, formatted_nodes: List[dict],):
        """
        Collect all nodes in the tree using depth-first traversal and format their data.
        
        :param node: The current TreeNode
        :param formatted_nodes: List to collect formatted node data
        """
        # ignore root node
        if node != root:
            formatted_nodes.append(node.format_node(root, img_path))
            
        for child in node.children:
            self._traverse_and_collect_formatted(root, child, img_path, formatted_nodes)
            
    def format_all_nodes(self) -> List[dict]:
        """
        Traverse the tree and format each node's data into the specified structure.
        
        :return: A list of dictionaries representing each node's formatted data
        """
        formatted_nodes = []
        self._traverse_and_collect_formatted(root=self.root, node=self.root, img_path=self.img_path, formatted_nodes=formatted_nodes)
        return formatted_nodes
    

def assigin_value_by_jsonl(input_file: str, output_file: str):
    """
    Read samples from a JSONL file, assign values to each sample, and write the output to a JSONL file.
    
    :param input_file: The input JSONL file containing samples
    :param output_file: The output JSONL file to write the valued samples
    """
    samples = read_samples_from_json(input_file, test=False)
    llava_value_sft_data_all = assign_value_for_samples(samples)
    write_samples_to_json(llava_value_sft_data_all, output_file)

def write_samples_to_json(llava_value_sft_data_all, output_file):
    with open(output_file, "w") as file:
        for d in llava_value_sft_data_all:
            file.write(json.dumps(d) + "\n")
            

def assign_value_for_samples(samples: List[dict]):
    """
    Assign values to each sample in the list.
    
    :param samples: A list of samples
    """
    sample_dict = {}
    for sample in samples:
        data_id = sample.get("data_id")
        if data_id is not None:
            if data_id not in sample_dict:
                sample_dict[data_id] = []
            sample_dict[data_id].append(sample)
    
    llava_value_sft_data_all = []
    for data_id, data_samples in sample_dict.items():
        tree = assign_value_by_data_piece(data_samples)
        tree.print_all_nodes()
        
        # ignore the sample with all false traces
        if tree.root.mk == math.inf:
            continue
        
        llava_value_data = tree.format_all_nodes()
        llava_value_sft_data_all.extend(llava_value_data)
    
    return llava_value_sft_data_all

    
    

def assign_value_by_data_piece(samples) -> Tree:
    """
    Assign values to each sample in the list.
    
    :param samples: A list of samples
    """
    
    tree_dict = {}
    
    # Note all the question and img_path of the samples are the same
    question = samples[0]["question"]
    img_path = samples[0].get("img_path", None)
    tree = Tree(question, img_path=img_path)
    tree_dict[question] = tree.root
    
    for sample in samples:
        for step_idx in range(1, len(sample["steps"])):
            # print(step_idx)
            # print(sample["steps"][step_idx])
            previous_steps = sample["steps"][:step_idx]
            previous_steps_str = '. '.join(previous_steps)
            current_steps = sample["steps"][:step_idx+1]
            current_steps_str = '. '.join(current_steps)

            current_step = sample["steps"][step_idx]
            
            # Note: only the first step (step_idx=0) shall be none
            previsou_node: TreeNode = tree_dict.get(previous_steps_str, None)
            # print(f'previous_steps_str: {previous_steps_str}')
            if step_idx > 1:
                assert previsou_node is not None
            
            # print(f'steps {sample["steps"]}')
            
            # print(f'tree {tree_dict}')
            previsou_node: TreeNode = tree_dict[previous_steps_str]
            if current_steps_str not in tree_dict:
                current_steps_str_without_q = '. '.join(sample["steps"][1:step_idx+1]) + '.'
                current_node = TreeNode(value=current_steps_str_without_q, step=current_step, correct=sample["correct"])
                previsou_node.add_child(current_node)
                tree_dict[current_steps_str] = current_node
                
    print(len(tree_dict))
    tree.update_tree()
    return tree
    
    
beam_sample_prompt = "Given the image, answer the following question. In your response, you should first state a step-by-step reasoning process, and conclude with your final answer from the given options.\n\nYour response must follow this format:\n\"Here is the step-by-step reasoning process:\n\nStep 1: ...\n\nStep 2: ...\n\nStep n: ...\n\nFinal Answer: ...\"\nHere is the question and options:\nQuestion: {question}\nOptions:\n{options}\n"


def generate_options_string(choices):
    # 获取字母序列 (A, B, C, ...)
    letters = string.ascii_uppercase
    # 检查是否足够选项字母
    if len(choices) > len(letters):
        raise ValueError("Too many choices for available letter options.")
    # 生成 "A. 选项内容" 的字符串列表
    options = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]
    # 将选项按行拼接为字符串
    return "\n".join(options)

def read_samples_from_json(input_file: str, test=False) -> List[dict]:
    """
    Read samples from a JSONL file and return a list of Sample objects.
    
    :param input_file: The input JSONL file containing samples
    :return: A list of Sample objects
    """
    samples = []
    if test:
        cnt = 0
    with open(input_file, "r") as file:
        for line in file:
            
            # # =======test=======
            # try:
            #     cnt+=1
            #     # if cnt < 1:
            #     #     continue
            #     # if cnt==1:
            #     #     print(line)
            #     if cnt > 2:
            #     # if cnt>1:
                    
            #         break
            # except:
            #     pass
            # # =======test=======
            
            
            data_raw = json.loads(line)
            
            #TODO: change options format, align with trace sample
            options = generate_options_string(data_raw["answer_choices"])
            question = beam_sample_prompt.format(question=data_raw["question"], options=options)
            
            if len(data_raw['result']) == 0:
                continue
            for res in data_raw['result']:
                value = res['value']
                if not value:
                    continue
                steps = get_steps_str_by_solution(value)
                assert len(steps)>0
                last_step = steps[-1]
                last_step_lower = last_step.lower()
                final_answer_lower = 'final answer'
                if final_answer_lower not in last_step_lower:
                    continue
                sample = {
                    "data_id": data_raw["data_id"],
                    "solution": value,
                    "question": question,
                    "img_path": data_raw['image_path'],
                    "correct": res['correct'],
                    "steps": [question]+steps
                }
            
                samples.append(sample)
    return samples


# def get_steps_str_by_solution(solution: str) -> List[str]:
#     """
#     从解决方案字符串中提取步骤。
    
#     :param solution: 包含步骤的解决方案字符串
#     :return: 步骤列表
#     """
#     # 定义正则表达式模式，匹配 "Step i" 或 "Final answer"
#     pattern = re.compile(r'(Step\s+\d+\.|Final answer\.?)', re.IGNORECASE)
    
#     # 查找所有匹配的位置
#     matches = list(pattern.finditer(solution))
    
#     steps = []
    
#     # 如果没有匹配到任何步骤标志，返回整个字符串作为一个步骤
#     if not matches:
#         return [solution.strip()]
    
#     for i in range(len(matches)):
#         start = matches[i].end()
#         if i + 1 < len(matches):
#             end = matches[i + 1].start()
#         else:
#             end = len(solution)
#         step_content = solution[start:end].strip(" .\n")
#         step_title = matches[i].group().rstrip('.')
#         full_step = f"{step_title}: {step_content}"
#         steps.append(full_step)
    
#     return steps


def get_steps_str_by_solution(solution: str) -> List[str]:
    """
    从解决方案字符串中提取独立的步骤。
    
    :param solution: 包含步骤的解决方案字符串
    :return: 步骤列表
    """
    steps = solution.strip().split(".")
    steps = [s for s in steps if s]
    
    print(steps)
    return steps
    # 定义正则表达式模式，匹配 "Step i:" 或 "Final answer:"，包括句点
    pattern = re.compile(r'(Step\s+\d+[:.]|Final answer[:.]?)', re.IGNORECASE)
    
    # 查找所有匹配的位置
    matches = list(pattern.finditer(solution))
    
    steps = []
    
    # 如果没有匹配到任何步骤标志，返回整个字符串作为一个步骤
    if not matches:
        return [solution.strip()]
    
    for i in range(len(matches)):
        # 当前步骤的开始位置
        start = matches[i].end()
        
        # 下一个步骤的开始位置，如果没有下一个，则为字符串末尾
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(solution)
        
        # 提取步骤内容，并去除前后的空格、句点和换行符
        step_content = solution[start:end].strip(" .\n")
        
        # 提取步骤标题，并去除末尾的句点
        step_title = matches[i].group().rstrip('.')
        
        # 组合步骤标题和内容
        full_step = f"{step_title}: {step_content}"
        
        # 添加到步骤列表中
        steps.append(full_step)
    
    return steps


def run_tree_test():
    # Construct the tree as per the test case

    # Root
    tree = Tree("Q: Roll two dice\n", 'img_path')

    # Level 1
    step1_1 = TreeNode(value="Q: Roll two dice\nStep 1_1: Total outcomes = 6*6=30", step="Step 1_1: Total outcomes = 6*6=30")
    step1_2 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36", step="Step 1_2: Total outcomes = 6*6=36")
    step1_3 = TreeNode("Q: Roll two dice\nStep 1_3: Probability = 1/6", step="Step 1_3: Probability = 1/6")

    tree.root.add_child(step1_1)
    tree.root.add_child(step1_2)
    tree.root.add_child(step1_3)

    # Level 2 LLaVA-REST-MCTS/utils/assign_value_tree.py
    step2_1 = TreeNode("Q: Roll two dice\nStep 1_1: Total outcomes = 6*6=30\nStep 2_1: List results (1,3), (2,2), (3,1), (4,0)", step="Step 2_1: List results (1,3), (2,2), (3,1), (4,0)")
    step2_2 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36\nStep 2_2: List results (1,3), (2,2), (3,1)", step="Step 2_2: List results (1,3), (2,2), (3,1)")
    step2_3 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36\nStep 2_3: List results (1,3), (2,2), (3,1)", step="Step 2_3: List results (1,3), (2,2), (3,1)")

    step1_1.add_child(step2_1)
    step1_2.add_child(step2_2)
    step1_2.add_child(step2_3)

    # Level 3
    step3_1 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36\nStep 2_2: List results (1,3), (2,2), (3,1)\nStep 3_1: Probability = 4/30=1/9", step="Step 3_1: Probability = 4/30=1/9")
    step3_2 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36\nStep 2_2: List results (1,3), (2,2), (3,1)\nStep 3_2: Probability = 3/36=1/12", step="Step 3_2: Probability = 3/36=1/12", correct=True)
    step3_3 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36\nStep 2_3: List results (1,3), (2,2), (3,1)\nStep 3_3: Probability = 3/36=1/12", step="Step 3_3: Probability = 3/36=1/12")

    step2_2.add_child(step3_1)
    step2_2.add_child(step3_2)
    step2_3.add_child(step3_3)

    # Final Level (Leaf Nodes)
    step4_1 = TreeNode("Q: Roll two dice\nStep 1_2: Total outcomes = 6*6=36\nStep 2_3: List results (1,3), (2,2), (3,1)\nStep 3_3: Probability = 3/36=1/12\nStep 4_1: Final probability = 3/36=1/12", step="Step 4_1: Final probability = 3/36=1/12", correct=True)

    step3_3.add_child(step4_1)

    # Now the entire tree is constructed. Call update_tree once to update all values.
    tree.update_tree()

    # Print all nodes with their attributes
    tree.print_all_nodes()
    llava_value_data = tree.format_all_nodes()
    
    print(llava_value_data)
    with open("llava_value_sft_data.jsonl", "w") as f:
        for node in llava_value_data:
            f.write(json.dumps(node) + "\n")

def test_solution_steps():
    solution_text = (
        "Step 1. 首先，我们需要确定问题的范围。 "
        "Step 2. 接下来，收集相关的数据。 "
        "Step 3. 分析数据以得出结论。 "
        "Final answer. 通过以上步骤，我们得出了最终的结论。"
    )
    
    solution_text = """Here is the step-by-step reasoning process:

Step 1: The people on the right are facing each other. Step 2: Since they are facing each other, it's likely that they are engaged in some kind of interaction. Step 3: The people on the right are talking. (This rationale supports Step 2 and provides more specific information about their activity.) Step 4: The people on the right are facing each other and talking. . ."""
    
    steps = get_steps_str_by_solution(solution_text)
    for step in steps:
        print(step)
    
    print(len(steps))
    
def test_assign_value_jsonl(input_path, output_path):
    assigin_value_by_jsonl(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Assign values to a JSONL file.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSONL file.')
    
    args = parser.parse_args()
    
    test_assign_value_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()




"""
Value: Q: Roll two dice
  Step: 
  Correct: False
  mk: 3
  correct_reachable: True
  rsk: 0.0
  wsk: 0.25
  vk: 0.0
----------------------------------------
Value: Step 1_1: Total outcomes = 6*6=30
  Step: 
  Correct: False
  mk: 3
  correct_reachable: False
  rsk: 1.0
  wsk: -0.25
  vk: 0.0
----------------------------------------
Value: Step 2_1: List results (1,3), (2,2), (3,1), (4,0)
  Step: 
  Correct: False
  mk: 3
  correct_reachable: False
  rsk: 1.0
  wsk: -0.25
  vk: 0.0
----------------------------------------
Value: Step 1_2: Total outcomes = 6*6=36
  Step: 
  Correct: False
  mk: 2
  correct_reachable: True
  rsk: 0.0
  wsk: 0.3333333333333333
  vk: 0.3333333333333333
----------------------------------------
Value: Step 2_2: List results (1,3), (2,2), (3,1)
  Step: 
  Correct: False
  mk: 1
  correct_reachable: True
  rsk: 0.0
  wsk: 0.33333333333333337
  vk: 0.6666666666666667
----------------------------------------
Value: Step 3_1: Probability = 4/30=1/9
  Step: 
  Correct: False
  mk: 1
  correct_reachable: False
  rsk: 1.0
  wsk: -0.16666666666666663
  vk: 0.5000000000000001
----------------------------------------
Value: Step 3_2: Probability = 3/36=1/12
  Step: 
  Correct: True
  mk: 0
  correct_reachable: True
  rsk: 0.0
  wsk: 0.33333333333333326
  vk: 1.0
----------------------------------------
Value: Step 2_3: List results (1,3), (2,2), (3,1)
  Step: 
  Correct: False
  mk: 2
  correct_reachable: True
  rsk: 0.0
  wsk: 0.22222222222222224
  vk: 0.5555555555555556
----------------------------------------
Value: Step 3_3: Probability = 3/36=1/12
  Step: 
  Correct: False
  mk: 1
  correct_reachable: True
  rsk: 0.0
  wsk: 0.2222222222222222
  vk: 0.7777777777777778
----------------------------------------
Value: Step 4_1: Final probability = 3/36=1/12
  Step: correct
  Correct: True
  mk: 0
  correct_reachable: True
  rsk: 0.0
  wsk: 0.2222222222222222
  vk: 1.0
----------------------------------------
Value: Step 1_3: Probability = 1/6
  Step: 
  Correct: False
  mk: 3
  correct_reachable: False
  rsk: 1.0
  wsk: -0.25
  vk: 0.0
----------------------------------------

"""