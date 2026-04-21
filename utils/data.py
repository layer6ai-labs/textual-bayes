import csv
import io
import json
import logging
import random
import tarfile
from pathlib import Path

import requests
from torch.utils.data import Dataset

import datasets as hf_datasets

log = logging.getLogger("Data")


class QaDataset(Dataset):
    def __len__(self):
        return len(self.data)


class ArcDataset(QaDataset):
    # todo: subset challenge and easy
    def __init__(self, split):
        raw_data = [
            hf_datasets.load_dataset("allenai/ai2_arc", "ARC-Easy"),
            hf_datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge"),
        ]
        self.data = hf_datasets.concatenate_datasets([subset[split] for subset in raw_data])

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        answers = example["choices"]["text"]
        label = ord(example["answerKey"]) - ord("A")
        return {"question": question, "answers": answers, "label": label}


class BoolqDataset(QaDataset):
    def __init__(self, split):
        raw_data = hf_datasets.load_dataset("google/boolq")
        self.data = raw_data[split]

    def __getitem__(self, idx):
        example = self.data[idx]
        context = example["passage"]
        question = example["question"]
        answers = ["True", "False"]
        label = 0 if example["answer"] else 1
        return {
            "context": context,
            "question": question,
            "answers": answers,
            "label": label,
        }


class CommonsenseqaDataset(QaDataset):
    def __init__(self, split):
        raw_data = hf_datasets.load_dataset("tau/commonsense_qa")
        self.data = raw_data[split]

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        answers = example["choices"]["text"]
        label = ord(example["answerKey"]) - ord("A")
        return {"question": question, "answers": answers, "label": label}


class CosmosqaDataset(QaDataset):
    # warning: test labels are not real
    def __init__(self, split, data_path="datasets/cosmosqa/data"):
        data_path = Path.cwd() / data_path
        self.data = []
        if split in ["train", "validation"]:
            with open(data_path / f"{split[:5]}.csv", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile, delimiter=",", quotechar='"')
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    self.data.append(row)
        else:
            with open(data_path / "test.jsonl", encoding="utf-8") as jsonfile:
                for line in jsonfile:
                    row = json.loads(line)
                    self.data.append(list(row.values()))
            with open(data_path / "sample_prediction.csv", encoding="utf-8") as csvfile:
                for i, line in enumerate(csvfile):
                    if i == 0:
                        continue
                    self.data[i - 1].append(line.strip().split(",")[1])

    def __getitem__(self, idx):
        example = self.data[idx]
        context = example[1]
        question = example[2]
        answers = example[3:7]
        label = int(example[7])
        return {
            "context": context,
            "question": question,
            "answers": answers,
            "label": label,
        }


class Gsm8kDataset(QaDataset):
    # todo: subset = [main|socratic]
    def __init__(self, split, subset="main"):
        raw_data = hf_datasets.load_dataset("openai/gsm8k", subset)
        self.data = raw_data[split]
        self.data = self.data.filter(
            lambda example: is_integer(
                str(example["answer"]).split("####")[1].strip().replace(",", "")
            )
        )

    def __getitem__(self, idx):
        example = self.data[idx]
        return example  # question and answer


class MmluDataset(QaDataset):
    # todo: topics
    def __init__(self, split):
        self.data = []
        raw_data = hf_datasets.load_dataset("cais/mmlu", "all")
        if split == "train":
            self.data = hf_datasets.concatenate_datasets(
                [raw_data["dev"], raw_data["auxiliary_train"]]
            )
        else:
            self.data = raw_data[split]

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        answers = example["choices"]
        label = example["answer"]
        return {"question": question, "answers": answers, "label": label}


class MmluProDataset(QaDataset):
    # todo: topics
    def __init__(self, split):
        self.data = []
        raw_data = hf_datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        self.data = raw_data[split]

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        answers = example["options"]
        label = example["answer_index"]
        return {"question": question, "answers": answers, "label": label}


class MultircDataset(QaDataset):
    # todo: it's labeled which sentences in the context are used for answering the question
    # but such information is not used in the current implementation
    def __init__(self, split, data_path="datasets/multirc"):
        data_path = Path.cwd() / data_path
        self.data = []
        filename = {
            "train": "train_456-fixedIds.json",
            "validation": "dev_83-fixedIds.json",
        }
        with open(data_path / filename[split], encoding="utf-8") as jsonfile:
            raw_data = json.load(jsonfile)["data"]
            for partition in raw_data:
                context = partition["paragraph"]["text"]
                for qa_pair in partition["paragraph"]["questions"]:
                    question = qa_pair["question"]
                    answers = []
                    labels = []
                    for i, answer in enumerate(qa_pair["answers"]):
                        answers.append(answer["text"])
                        if answer["isAnswer"]:
                            labels.append(i)
                    self.data.append(
                        {
                            "context": context,
                            "question": question,
                            "answers": answers,
                            "label": labels,
                        }
                    )
        self.len = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class OpenbookqaDataset(QaDataset):
    # todo: subset main and additional, where additional has extra info that's not loaded
    def __init__(self, split):
        raw_data = [
            hf_datasets.load_dataset("allenai/openbookqa", "main"),
            hf_datasets.load_dataset("allenai/openbookqa", "additional"),
        ]
        self.data = hf_datasets.concatenate_datasets([subset[split] for subset in raw_data])

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question_stem"]
        answers = example["choices"]["text"]
        label = ord(example["answerKey"]) - ord("A")
        return {"question": question, "answers": answers, "label": label}


_QASPER_URLS = {
    "train": (
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "qasper-train-v0.3.json",
    ),
    "validation": (
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "qasper-dev-v0.3.json",
    ),
    "test": (
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
        "qasper-test-v0.3.json",
    ),
}


def _load_qasper_json(split):
    """Load qasper from S3 directly, bypassing the deprecated HF dataset script."""
    url, filename = _QASPER_URLS[split]
    cache_dir = Path.home() / ".cache" / "qasper"
    cache_file = cache_dir / filename

    if not cache_file.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
            f = tar.extractfile(filename)
            raw = json.loads(f.read().decode("utf-8"))
        with open(cache_file, "w") as f:
            json.dump(raw, f)
    else:
        with open(cache_file) as f:
            raw = json.load(f)

    # Convert raw JSON (dict of paper_id → paper) to a list of paper dicts matching
    # the structure that HF datasets Sequence would produce (list-of-dicts → dict-of-lists).
    papers = []
    for paper_id, paper in raw.items():
        qas_raw = paper.get("qas", [])
        qas_hf = {
            "question": [qa["question"] for qa in qas_raw],
            "answers": [
                {
                    "answer": [ann["answer"] for ann in qa.get("answers", [])],
                    "annotation_id": [
                        ann.get("annotation_id", "") for ann in qa.get("answers", [])
                    ],
                    "worker_id": [ann.get("worker_id", "") for ann in qa.get("answers", [])],
                }
                for qa in qas_raw
            ],
        }
        papers.append({**paper, "id": paper_id, "qas": qas_hf})
    return papers


class QasperDataset(QaDataset):
    def __init__(self, split="train", question_type="freeform"):
        """
        Args:
            split: specify the splits using; Options: [train|test|all]
            question_type: specify what type of questions you want to include; Options:[freeform|selection|extractive|all]
        """
        # train 888 papers, 2593 questions; test 416 papers; all 1585 papers
        raw_data = _load_qasper_json(split)
        self.data = self.get_datapoints(raw_data, question_type, use_random_context=False)

    def get_datapoints(self, split_data, question_type, use_random_context=False):
        """
        Args:
            split_data: the raw data loaded from the dataset
            question_type: specify what type of questions you want to include; Options:[freeform|selection|extractive|all]
            use_random_context: whether to randomly select a context for the unanswerable questions
        """
        if use_random_context:
            # fix the random seed for reproducibility
            random.seed(42)
            # collect all the contexts to sample from
            all_contexts = []
            for sample in split_data:
                # sample is a paper and all questions and answers attached to it
                qa = sample["qas"]
                answers = qa["answers"]
                for gt_annotation in answers:
                    answer_list = gt_annotation["answer"]
                    for i, answer_info in enumerate(answer_list):
                        if i == 0:
                            context = " ".join(answer_info["evidence"])
                            if context != "":
                                all_contexts.append(context)

        num_filtered_qa = 0
        data_list = []
        for sample in split_data:
            # sample is a paper and all questions and answers attached to it
            qa = sample["qas"]
            # multiple question
            questions = qa["question"]
            answers = qa["answers"]
            for question, answer_dict in zip(questions, answers):
                answer_gt = None
                context = []
                unanswerable_flag = False
                skip_flag = False
                answer_list = answer_dict["answer"]
                for i, answer_info in enumerate(answer_list):
                    if i == 0:
                        # take the first answer following the repo of llmscienceqa
                        context = " ".join(answer_info["evidence"])
                        if answer_info["unanswerable"]:
                            unanswerable_flag = answer_info["unanswerable"]
                            answer_gt = None
                            if use_random_context:
                                # randomly select a context for the unanswerable questions
                                context = random.choice(all_contexts)
                        else:
                            if answer_info["yes_no"] is not None:
                                # selection question
                                qtype = "selection"
                                if question_type in ["all", "selection"]:
                                    if answer_info["yes_no"]:
                                        answer_gt = "yes"
                                    else:
                                        answer_gt = "no"
                                else:
                                    skip_flag = True

                            if answer_info["free_form_answer"]:
                                # free form question
                                qtype = "freeform"
                                if question_type in ["all", "freeform"]:
                                    answer_gt = answer_info["free_form_answer"]
                                else:
                                    skip_flag = True

                            if answer_info["extractive_spans"]:
                                # extractive question
                                qtype = "extractive"
                                if question_type in ["all", "extractive"]:
                                    if isinstance(answer_info["extractive_spans"], list):
                                        answer_gt = " ,".join(answer_info["extractive_spans"])
                                    else:
                                        answer_gt = answer_info["extractive_spans"]
                                else:
                                    skip_flag = True

                if "FLOAT SELECTED" not in context and not skip_flag:
                    # if the evidence (context) doesn't include tables or figures
                    data_list.append(
                        {
                            "context": context,
                            "question": question,
                            "question_type": qtype,
                            "answer": answer_gt,
                            "unanswerable": unanswerable_flag,
                        }
                    )
                else:
                    num_filtered_qa += 1
        log.info(
            f"filtered out {num_filtered_qa} questions with table/figure evidence or type of questions"
        )
        return data_list

    def __getitem__(self, idx):
        """
        Return:
            dict: {
                "context": context,
                "question": question,
                "question_type: ["freeform"|"selection"|"extractive"],
                "answer": answer,
                "unanswerable: [True|False]
            }
        """
        return {
            **self.data[idx],
            "qid": idx,
        }


class QasperRandomDataset(QasperDataset):
    def __init__(self, split="train", question_type="freeform"):
        """
        Args:
            split: specify the splits using; Options: [train|test|all]
            question_type: specify what type of questions you want to include; Options:[freeform|selection|extractive|all]
        """
        # train 888 papers, 2593 questions; test 416 papers; all 1585 papers
        raw_data = hf_datasets.load_dataset("allenai/qasper", split=split)
        self.data = self.get_datapoints(raw_data, question_type, use_random_context=True)


class PubMedQADataset(QaDataset):
    def __init__(self, split="train"):
        raw_data = hf_datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
        self.data = []
        for dp in raw_data:
            if dp["final_decision"] == "maybe":
                unanswerable_flag = True
                answer = "maybe"
            else:
                unanswerable_flag = False
                answer = dp["final_decision"]
            self.data.append(
                {
                    "question": dp["question"],
                    "context": "\n".join(dp["context"]["contexts"]),
                    "answer": answer,
                    "unanswerable": unanswerable_flag,
                }
            )

    def __getitem__(self, idx):
        """
        Return:
            dict: {
                "context": context,
                "question": question,
                "answer": answer, ('yes', 'no', 'maybe')
                "unanswerable: [True|False]
            }
        """
        return self.data[idx]


class GpqaDataset(QaDataset):
    def __init__(self, split, seed=42):
        random.seed(seed)
        raw_data = hf_datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        self.data = raw_data["train"].train_test_split(test_size=100, seed=seed)
        self.data = self.data[split]
        self.data = self.data.map(self.process)

    def process(self, example):
        list_choices = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
            example["Correct Answer"],
        ]
        random.shuffle(list_choices)
        example["Choices"] = list_choices
        example["Label"] = list_choices.index(example["Correct Answer"])
        return example

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["Question"]
        answers = example["Choices"]
        label = example["Label"]
        return {"question": question, "answers": answers, "label": label}


class AIMEDataset(QaDataset):
    def __init__(self, split):
        test_raw_data = hf_datasets.load_dataset("Maxwell-Jia/AIME_2024")
        filtered_test_raw_data = test_raw_data["train"].filter(
            lambda example: is_integer(str(example["Answer"]))
        )

        raw_data = hf_datasets.load_dataset("di-zhang-fdu/AIME_1983_2024")
        filtered_raw_data = raw_data["train"].filter(
            lambda example: (example["Year"] < 2024) and (example["Year"] >= 2020)
        )
        filtered_train_raw_data = filtered_raw_data.filter(
            lambda example: is_integer(str(example["Answer"]))
        )
        self.data = {
            "test": filtered_test_raw_data,
            "train": filtered_train_raw_data,
        }
        self.data = self.data[split]

    def __getitem__(self, idx):
        example = self.data[idx]
        if "Question" in example:
            question = example["Question"]
        else:
            question = example["Problem"]
        answer = example["Answer"]
        # leave the Solution (Reasoning trajectory) out
        return {
            "question": question,
            "answer": answer,
        }  # question and answer


class FactScoreDataset(QaDataset):
    def __init__(self, path="datasets/fact_score", split="train"):
        data_path = Path.cwd() / path
        self.data = []
        filename = f"factscore_{split}.jsonl"
        with open(data_path / filename, encoding="utf-8") as jsonfile:
            for line in jsonfile:
                row = json.loads(line)
                question = row["question"]
                self.data.append(
                    {
                        "question": question,
                    }
                )

    def __getitem__(self, idx):
        """
        Return:
            dict: {
                "question": question,
            }
        """
        return self.data[idx]


class SimpleqaDataset(QaDataset):
    def __init__(self, split, seed=42):
        random.seed(seed)
        raw_data = hf_datasets.load_dataset("basicv8vc/SimpleQA")["test"]
        raw_data = raw_data.train_test_split(test_size=100, seed=seed)
        self.data = raw_data[split]

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "question": example["problem"],
            "answer": example["answer"],
        }


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


dataset_classes = {
    "arc": ArcDataset,
    "boolq": BoolqDataset,
    "commonsenseqa": CommonsenseqaDataset,
    "cosmosqa": CosmosqaDataset,
    "gsm8k": Gsm8kDataset,
    "aime": AIMEDataset,
    "mmlu": MmluDataset,
    "multirc": MultircDataset,
    "openbookqa": OpenbookqaDataset,
    "qasper": QasperDataset,
    "qasper-random": QasperRandomDataset,
    "pubmedqa": PubMedQADataset,
    "mmlu-pro": MmluProDataset,
    "gpqa": GpqaDataset,
    "factscore": FactScoreDataset,
    "simpleqa": SimpleqaDataset,
}
