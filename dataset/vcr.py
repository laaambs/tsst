import jsonlines
import os
from tqdm import tqdm
import json
from copy import deepcopy
import yaml
from torch.utils.data import Dataset, DataLoader

import random
class VCRSampler:
    def __init__(
        self, dataset_root, dataset_name, data_subset, data_partition, caption_path, seed=42
    ):
        self.dataset_name = dataset_name
        self.ann = {}
        if "val" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "val.jsonl")
        elif "train" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "train.jsonl")
        elif "test_gen" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "test_gen.jsonl")
        # Obtain subset annot_ids
        id_partitions = None
        if data_subset is not None:
            with open(data_subset, "r") as file:
                id_partitions = yaml.safe_load(file)

        with jsonlines.open(dataset_anno_dir) as reader:
            for cur_ann in tqdm(reader):
                annot_id = cur_ann["annot_id"]
                # Only keep the input partition.
                if id_partitions is not None:
                    if annot_id not in id_partitions:
                        continue

                img_path = os.path.join(dataset_root, "vcr1images", cur_ann["img_fn"])
                meta_path = os.path.join(
                    dataset_root, "vcr1images", cur_ann["metadata_fn"]
                )

                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                # TODO: which one is correct?
                obj_names = cur_ann["objects"]
                obj_boxes = metadata["boxes"]
                obj_width = metadata["width"]
                new_obj_names = self.add_person_location(
                    cur_ann["question"],
                    cur_ann["answer_choices"],
                    obj_names,
                    obj_boxes,
                    obj_width,
                )

                question_sent = self.transform_list2sent(
                    cur_ann["question"], new_obj_names
                )

                answer_sents = []
                # qa_sent = []
                for answer_i in cur_ann["answer_choices"]:
                    answer_sents.append(
                        self.transform_list2sent(answer_i, new_obj_names)
                    )
                    
                answer_choices = '\n'.join(answer_sents)
                question_with_choices = f"""{question_sent}. Choices:\n{answer_choices}"""


                # TODO: Add data loading for rationale when doing QA2R task.
                self.ann[annot_id] = {
                    "question": question_sent,
                    "question_with_choices": question_with_choices,
                    "answer_choices": answer_sents,
                    #   'rationale_choices': cur_ann['rationale_choices'],
                    "answer_str": answer_sents[cur_ann["answer_label"]],
                    "img_path": img_path,
                    "meta_path": meta_path,
                    "processed_img_path": img_path,
                }

                if "val" in dataset_name or "train" in dataset_name:
                    self.ann[annot_id]["answer_label"] = cur_ann["answer_label"] + 1
                    # self.ann[annot_id]['rationale_label'] = cur_ann['rationale_label']

                # Load caption if any.
                if caption_path is not None:
                    caption_file = os.path.join(
                        caption_path, "{}.yaml".format(annot_id)
                    )
                    if os.path.isfile(caption_file):
                        with open(caption_file, "r") as f:
                            cur_cap = yaml.safe_load(f)
                        assert cur_cap["id"] == annot_id
                        self.ann[annot_id]["caption"] = cur_cap["new_caption"]
                    else:
                        self.ann[annot_id]["caption"] = None

        # Only keep samples in partitions.
        if data_partition is not None:
            start_data_id, end_data_id = data_partition.split("_")
            sorted_ids = sorted(list(self.ann.keys()))
            random.seed(seed)  
            random.shuffle(sorted_ids)
            subset_ids = sorted_ids[
                int(start_data_id) : int(end_data_id) + 1
            ]
            self.ann = {key: self.ann[key] for key in subset_ids}

        self._ids = random.shuffle(sorted(list(self.ann.keys())))

    def add_person_location(self, questions, answers, obj_names, obj_boxes, obj_width):
        referred_person_id = []
        referred_person_rename = []
        referred_person_coor = []

        left_range = [0, obj_width / 3]
        middle_range = [obj_width / 3, obj_width * 2 / 3]
        right_range = [obj_width * 2 / 3, obj_width]

        for ii in questions:
            if isinstance(ii, list) and obj_names[ii[0]] == "person":
                if ii[0] not in referred_person_id:
                    referred_person_id.append(ii[0])
                    referred_person_rename.append("person")
                    referred_person_coor.append(
                        (obj_boxes[ii[0]][0] + obj_boxes[ii[0]][2]) / 2
                    )
        for ans_i in answers:
            for ii in ans_i:
                if isinstance(ii, list) and obj_names[ii[0]] == "person":
                    if ii[0] not in referred_person_id:
                        referred_person_id.append(ii[0])
                        referred_person_rename.append("person")
                        referred_person_coor.append(
                            (obj_boxes[ii[0]][0] + obj_boxes[ii[0]][2]) / 2
                        )

        if len(referred_person_id) == 0:
            # Don't make change.
            return obj_names
        else:
            if len(referred_person_id) == 1:
                cur_person_id = referred_person_id[0]
                if left_range[0] <= obj_boxes[cur_person_id][0] <= left_range[1]:
                    referred_person_rename[0] = "person on the left"
                elif middle_range[0] < obj_boxes[cur_person_id][0] <= middle_range[1]:
                    referred_person_rename[0] = "person in the middle"
                elif right_range[0] < obj_boxes[cur_person_id][0] <= right_range[1]:
                    referred_person_rename[0] = "person on the right"
            elif len(referred_person_id) == 2:
                left_right_id = sorted(
                    range(len(referred_person_coor)),
                    key=lambda k: referred_person_coor[k],
                )
                referred_person_rename[left_right_id[0]] = "person on the left"
                referred_person_rename[left_right_id[1]] = "person on the right"
            elif len(referred_person_id) == 3:
                left_right_id = sorted(
                    range(len(referred_person_coor)),
                    key=lambda k: referred_person_coor[k],
                )
                referred_person_rename[left_right_id[0]] = "person on the left"
                referred_person_rename[left_right_id[1]] = "person in the middle"
                referred_person_rename[left_right_id[2]] = "person on the right"
            else:
                for box_id, box_coor in enumerate(referred_person_coor):
                    if left_range[0] <= box_coor <= left_range[1]:
                        referred_person_rename[box_id] = (
                            "person on the left"
                            if "person on the left" not in referred_person_rename
                            else "another person on the left"
                        )
                    elif middle_range[0] < box_coor <= middle_range[1]:
                        referred_person_rename[box_id] = (
                            "person in the middle"
                            if "person in the middle" not in referred_person_rename
                            else "another person in the middle"
                        )
                    elif right_range[0] < box_coor <= right_range[1]:
                        referred_person_rename[box_id] = (
                            "person  on the right"
                            if "person on the right" not in referred_person_rename
                            else "another person on the right"
                        )

            for person_id, person_real_id in enumerate(referred_person_id):
                obj_names[person_real_id] = referred_person_rename[person_id]
            return obj_names

    def transform_list2sent(self, input, objs):
        try:
            input_sent = [objs[ii[0]] if isinstance(ii, list) else ii for ii in input]
        except:
            print("???")
        input_sent = (" ").join(input_sent)
        input_sent = input_sent.replace(" ,", ",")
        input_sent = input_sent.replace(" .", ".")
        input_sent = input_sent.replace(" ?", "?")
        return input_sent

    @property
    def ids(self):
        return deepcopy(self._ids)

    def fetch_data(self, id):
        ann = self.ann[id]
        img_path = ann["img_path"]

        return img_path, ann
    
    
    
class VCRDataset(Dataset):
    def __init__(
        self, dataset_root, dataset_name, data_subset=None, data_partition=None, caption_path=None
    ):
        self.dataset_name = dataset_name
        self.ann = {}

        # 确定数据集路径
        if "val" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "val.jsonl")
        elif "train" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "train.jsonl")
        elif "test_gen" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "test_gen.jsonl")
        elif "test" in dataset_name:
            dataset_anno_dir = os.path.join(dataset_root, "test.jsonl")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        # 读取子集 id（可选）
        id_partitions = None
        if data_subset is not None:
            with open(data_subset, "r") as file:
                id_partitions = yaml.safe_load(file)

        # 解析数据集
        with jsonlines.open(dataset_anno_dir) as reader:
            for cur_ann in tqdm(reader):
                annot_id = cur_ann["annot_id"]

                # 仅保留指定的分区 id
                if id_partitions is not None and annot_id not in id_partitions:
                    continue

                img_path = os.path.join(dataset_root, "vcr1images", cur_ann["img_fn"])
                meta_path = os.path.join(
                    dataset_root, "vcr1images", cur_ann["metadata_fn"]
                )

                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                obj_names = cur_ann["objects"]
                obj_boxes = metadata["boxes"]
                obj_width = metadata["width"]

                new_obj_names = self.add_person_location(
                    cur_ann["question"],
                    cur_ann["answer_choices"],
                    obj_names,
                    obj_boxes,
                    obj_width,
                )

                question_sent = self.transform_list2sent(
                    cur_ann["question"], new_obj_names
                )

                answer_sents = [
                    self.transform_list2sent(answer, new_obj_names)
                    for answer in cur_ann["answer_choices"]
                ]
                answer_sents_str = "\n".join(answer_sents)

                question_with_choices = f"""{question_sent}. Choices:\n{answer_sents_str}"""

                

                self.ann[annot_id] = {
                    "question": question_sent,
                    "question_with_choices": question_with_choices,
                    "answer_choices": answer_sents,
                    "answer_str": answer_sents[cur_ann["answer_label"]],
                    "img_path": img_path,
                    "meta_path": meta_path,
                    "processed_img_path": img_path,
                }

                if "val" in dataset_name or "train" in dataset_name:
                    self.ann[annot_id]["answer_label"] = cur_ann["answer_label"] + 1

                # 加载描述信息（可选）
                if caption_path is not None:
                    caption_file = os.path.join(
                        caption_path, f"{annot_id}.yaml"
                    )
                    if os.path.isfile(caption_file):
                        with open(caption_file, "r") as f:
                            cur_cap = yaml.safe_load(f)
                        assert cur_cap["id"] == annot_id
                        self.ann[annot_id]["caption"] = cur_cap["new_caption"]
                    else:
                        self.ann[annot_id]["caption"] = None

        # 根据数据分区筛选样本
        if data_partition is not None:
            start_data_id, end_data_id = data_partition.split("_")
            subset_ids = list(self.ann.keys())[
                int(start_data_id): int(end_data_id) + 1
            ]
            self.ann = {key: self.ann[key] for key in subset_ids}

        self._ids = list(self.ann.keys())

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, index):
        annot_id = self._ids[index]
        ann = self.ann[annot_id]
        
        
        
        return {
            "id": annot_id,
            "question": ann["question"],
            "question_with_choices": ann["question_with_choices"],
            "answer_choices": ann["answer_choices"],
            "answer_str": ann["answer_str"],
            "img_path": ann["img_path"],
            "meta_path": ann["meta_path"],
            "processed_img_path": ann["processed_img_path"],
            "answer_label": ann.get("answer_label"),
            "caption": ann.get("caption"),
        }

    def add_person_location(self, questions, answers, obj_names, obj_boxes, obj_width):
        referred_person_id = []
        referred_person_rename = []
        referred_person_coor = []

        left_range = [0, obj_width / 3]
        middle_range = [obj_width / 3, obj_width * 2 / 3]
        right_range = [obj_width * 2 / 3, obj_width]

        for ii in questions:
            if isinstance(ii, list) and obj_names[ii[0]] == "person":
                if ii[0] not in referred_person_id:
                    referred_person_id.append(ii[0])
                    referred_person_rename.append("person")
                    referred_person_coor.append(
                        (obj_boxes[ii[0]][0] + obj_boxes[ii[0]][2]) / 2
                    )
        for ans_i in answers:
            for ii in ans_i:
                if isinstance(ii, list) and obj_names[ii[0]] == "person":
                    if ii[0] not in referred_person_id:
                        referred_person_id.append(ii[0])
                        referred_person_rename.append("person")
                        referred_person_coor.append(
                            (obj_boxes[ii[0]][0] + obj_boxes[ii[0]][2]) / 2
                        )

        if len(referred_person_id) == 0:
            return obj_names
        else:
            if len(referred_person_id) == 1:
                cur_person_id = referred_person_id[0]
                if left_range[0] <= obj_boxes[cur_person_id][0] <= left_range[1]:
                    referred_person_rename[0] = "person on the left"
                elif middle_range[0] < obj_boxes[cur_person_id][0] <= middle_range[1]:
                    referred_person_rename[0] = "person in the middle"
                elif right_range[0] < obj_boxes[cur_person_id][0] <= right_range[1]:
                    referred_person_rename[0] = "person on the right"
            elif len(referred_person_id) == 2:
                left_right_id = sorted(
                    range(len(referred_person_coor)),
                    key=lambda k: referred_person_coor[k],
                )
                referred_person_rename[left_right_id[0]] = "person on the left"
                referred_person_rename[left_right_id[1]] = "person on the right"
            elif len(referred_person_id) == 3:
                left_right_id = sorted(
                    range(len(referred_person_coor)),
                    key=lambda k: referred_person_coor[k],
                )
                referred_person_rename[left_right_id[0]] = "person on the left"
                referred_person_rename[left_right_id[1]] = "person in the middle"
                referred_person_rename[left_right_id[2]] = "person on the right"
            else:
                for box_id, box_coor in enumerate(referred_person_coor):
                    if left_range[0] <= box_coor <= left_range[1]:
                        referred_person_rename[box_id] = (
                            "person on the left"
                            if "person on the left" not in referred_person_rename
                            else "another person on the left"
                        )
                    elif middle_range[0] < box_coor <= middle_range[1]:
                        referred_person_rename[box_id] = (
                            "person in the middle"
                            if "person in the middle" not in referred_person_rename
                            else "another person in the middle"
                        )
                    elif right_range[0] < box_coor <= right_range[1]:
                        referred_person_rename[box_id] = (
                            "person on the right"
                            if "person on the right" not in referred_person_rename
                            else "another person on the right"
                        )

            for person_id, person_real_id in enumerate(referred_person_id):
                obj_names[person_real_id] = referred_person_rename[person_id]
            return obj_names

    def transform_list2sent(self, input, objs):
        try:
            input_sent = [objs[ii[0]] if isinstance(ii, list) else ii for ii in input]
        except:
            print("Error while transforming list to sentence.")
        input_sent = (" ").join(input_sent)
        input_sent = input_sent.replace(" ,", ",")
        input_sent = input_sent.replace(" .", ".")
        input_sent = input_sent.replace(" ?", "?")
        return input_sent
    
