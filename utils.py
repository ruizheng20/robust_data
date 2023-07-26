# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : utils.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 19:00
@Author   : Zhiheng Xi
"""

import logging
import datasets
from torch.utils.data import Dataset

import torch
from torch.nn.utils.rnn import pad_sequence
import random
import os
task_to_keys = {
    "ag_news": ("text", None),
    "imdb": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "SetFit/20_newsgroups":("text",None)
}
def dataset_script_path(name):
    if name=="glue":
        return "=dataset_scripts/glue"
    elif name=="imdb":
        return "dataset_scripts/imdb"
    elif name=="ag_news":
        return "dataset_scripts/ag_news"
    else:
        return name
def dataset_to_length_and_batch_size(dataset_name="glue",task_name="sst2"):
    if dataset_name == "imdb":
        len_dataset = 25000
        per_device_batch_size = 32
        return len_dataset,per_device_batch_size
    elif dataset_name == "ag_news":
        len_dataset = 120000
        per_device_batch_size = 32
        return len_dataset,per_device_batch_size

    elif dataset_name == "glue":
        if task_name == "sst2" or task_name == "sst-2":
            len_dataset = 67349
            per_device_batch_size = 32
            return len_dataset, per_device_batch_size
        elif task_name == "qnli":
            len_dataset = 104743
            per_device_batch_size = 32
            return len_dataset, per_device_batch_size
        elif task_name == "qqp":
            len_dataset = 363846
            per_device_batch_size = 32
            return len_dataset, per_device_batch_size
        elif task_name == "mnli":
            len_dataset = 392702
            per_device_batch_size = 32
            return len_dataset, per_device_batch_size

MAX_CONTEXT_LEN = 50
MAX_SEQ_LEN = 128
logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x,i=1):
        self._x += x
        self._i += i

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x / (self._i + 1e-13)


class Collator:
    """
    Collates transformer outputs.
    """

    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels = list(zip(*features))
        # Assume that all inputs have the same keys as the first
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        labels = torch.tensor([x for x in labels])
        return padded_inputs, labels


class Huggingface_dataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            name_or_dataset: str,
            subset: str = None,
            split="train",
            shuffle=False,

    ):

        self.args = args
        if self.args.max_seq_length is None:
            self.args.max_seq_length = MAX_SEQ_LEN
        self.tokenizer = tokenizer
        self.name = name_or_dataset
        self.subset = subset
        if self.name == "SetFit/20_newsgroups":
            self.dataset = datasets.load_dataset(dataset_script_path(self.name),subset)["train"]
            total_size = len(self.dataset)
            train_size = int(total_size * 0.85)
            test_size = total_size - train_size
            self.dataset = self.dataset.train_test_split(total_size-train_size,train_size,seed=42)
            if split=="train":
                self.dataset = self.dataset["train"]
            else:
                self.dataset = self.dataset["test"]

        else:
            self.dataset = datasets.load_dataset(dataset_script_path(self.name), subset)
            self.dataset = self.dataset[split]
        if subset is not None:
            self.input_columns = task_to_keys[subset]
        else:
            self.input_columns = task_to_keys[name_or_dataset]
        self.key1 = self.input_columns[0]
        self.key2 = self.input_columns[1]
        self.shuffled = shuffle

        if shuffle:
            self.dataset.shuffle()
        if self.args.random_select and split=="train" :
            if self.args.data_indices_file != None and os.path.exists(self.args.data_indices_file):
                with open(self.args.data_indices_file, 'r') as f:
                    data_indices = f.readlines()[0][1:-1].split(",")
                    data_indices = [int(e) for e in data_indices]
                    # print(data_indices)
            else:
                data_indices = self.generate_random_indices()
                data_indices = sorted(data_indices)
                with open(self.args.data_indices_file,'w') as f:
                    f.writelines(str(data_indices))
            self.dataset = self.dataset.select(data_indices)

            # data_indices_output_file = os.path.join(args.output_dir, 'data_indices_file_ratio{}'.format(self.args.num_train_examples_ratio))
            # with open(output_file,'w') as f:
            #     f.writelines(str(data_indices))

    def generate_random_indices(self):
        data_indices = set([])
        num_train_examples_per_class = self.args.num_train_examples_ratio * len(self.dataset) // self.args.num_labels
        random_val = random.randint(0,1)
        selected_label_indices = {}
        for i in range(self.args.num_labels):
            selected_label_indices[i] = 0
        while True:
            random_val = random.randint(0,len(self.dataset)-1)
            if random_val not in data_indices and selected_label_indices[self.dataset[random_val]['label']] < num_train_examples_per_class:
                data_indices.add(random_val)
                selected_label_indices[self.dataset[random_val]['label']] += 1

            if len(data_indices) >= num_train_examples_per_class * self.args.num_labels:
                break

        return list(data_indices)



    def _format_examples(self, examples):
        """
        Only for some task which has ONE input column, such as SST-2 and IMDB, NOT work for NLI such as MRPC.
        """

        texts = ((examples[self.key1],) if self.key2 is None else (examples[self.key1], examples[self.key2]))
        inputs = self.tokenizer(*texts, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt')

        # text2 = [examples[self.input_columns[0]]]
        # sentence2 = "".join(text2)
        # inputs2 = self.tokenizer(sentence2, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt')

        # output = [int(examples['label']),index]
        output = int(examples['label'])
        return (inputs, output)

    def shuffle(self):
        self.dataset.shuffle()
        self.shuffled = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            # return self._format_examples(self.dataset[i])
            return self._format_examples(self.dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [
                self._format_examples(self.dataset[j]) for j in range(i.start, i.stop)
            ]

class Huggingface_dataset_with_data_id(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            name_or_dataset: str,
            subset: str = None,
            split="train",
            shuffle=False,

    ):

        self.args = args
        if self.args.max_seq_length is None:
            self.args.max_seq_length = MAX_SEQ_LEN
        self.tokenizer = tokenizer
        self.name = name_or_dataset
        self.subset = subset
        if self.name == "SetFit/20_newsgroups":
            self.dataset = datasets.load_dataset(dataset_script_path(self.name),subset)["train"]
            total_size = len(self.dataset)
            train_size = int(total_size * 0.85)
            test_size = total_size - train_size
            self.dataset = self.dataset.train_test_split(total_size-train_size,train_size,seed=42)
            if split=="train":
                self.dataset = self.dataset["train"]
            else:
                self.dataset = self.dataset["test"]

        else:
            self.dataset = datasets.load_dataset(dataset_script_path(self.name), subset)
            self.dataset = self.dataset[split]

        if subset is not None:
            self.input_columns = task_to_keys[subset]
        else:
            self.input_columns = task_to_keys[name_or_dataset]
        self.key1 = self.input_columns[0]
        self.key2 = self.input_columns[1]
        self.shuffled = shuffle

        if shuffle:
            self.dataset.shuffle()
        if self.args.random_select and split=="train" :
            if self.args.data_indices_file != None and os.path.exists(self.args.data_indices_file):
                with open(self.args.data_indices_file, 'r') as f:
                    data_indices = f.readlines()[0][1:-1].split(",")
                    data_indices = [int(e) for e in data_indices]
                    # print(data_indices)
            else:
                data_indices = self.generate_random_indices()
                data_indices = sorted(data_indices)
                if self.args.data_indices_file!=None:
                    with open(self.args.data_indices_file,'w') as f:
                        f.writelines(str(data_indices))
            self.dataset = self.dataset.select(data_indices)

            # data_indices_output_file = os.path.join(args.output_dir, 'data_indices_file_ratio{}'.format(self.args.num_train_examples_ratio))
            # with open(output_file,'w') as f:
            #     f.writelines(str(data_indices))

    def generate_random_indices(self):
        data_indices = set([])
        num_train_examples_per_class = self.args.num_train_examples_ratio * len(self.dataset) // self.args.num_labels
        random_val = random.randint(0,1)
        selected_label_indices = {}
        for i in range(self.args.num_labels):
            selected_label_indices[i] = 0
        while True:
            random_val = random.randint(0,len(self.dataset)-1)
            if random_val not in data_indices and selected_label_indices[self.dataset[random_val]['label']] < num_train_examples_per_class:
                data_indices.add(random_val)
                selected_label_indices[self.dataset[random_val]['label']] += 1

            if len(data_indices) >= num_train_examples_per_class * self.args.num_labels:
                break

        return list(data_indices)

    def updata_idx(self,data_indices_to_new_id,new_id_to_data_indices):
        self.data_indices_to_new_id = data_indices_to_new_id
        self.new_id_to_data_indices = new_id_to_data_indices

    def _format_examples(self, examples,i):
        """
        Only for some task which has ONE input column, such as SST-2 and IMDB, NOT work for NLI such as MRPC.
        """

        texts = ((examples[self.key1],) if self.key2 is None else (examples[self.key1], examples[self.key2]))
        inputs = self.tokenizer(*texts, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt')

        # text2 = [examples[self.input_columns[0]]]
        # sentence2 = "".join(text2)
        # inputs2 = self.tokenizer(sentence2, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt')
        # from baselines.proposed_method.new_fine_tune import data_indices_to_new_id,new_id_to_data_indices

        if (self.name=="imdb" or self.name=="ag_news"):
            if len(self.dataset)<dataset_to_length_and_batch_size(self.name,self.subset)[0]:

                output = [int(examples['label']),self.new_id_to_data_indices[i]]
            else:
                output = [int(examples['label']), i]
        else:
            output = [int(examples['label']),examples["idx"]]
        # output = int(examples['label'])
        return (inputs, output)

    def shuffle(self):
        self.dataset.shuffle()
        self.shuffled = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            # return self._format_examples(self.dataset[i])
            return self._format_examples(self.dataset[i],i)
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [
                self._format_examples(self.dataset[j]) for j in range(i.start, i.stop)
            ]

class local_dataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            name_or_dataset: str,
            subset: str = 'sst2',
            split="train",
            data_type="csv",
            shuffle=False,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.type = data_type
        self.name = name_or_dataset
        self.subset = subset
        self.dataset = datasets.load_dataset(self.type, data_files=self.name)[split]
        if subset is not None:
            self.input_columns = task_to_keys[subset]
        else:
            self.input_columns = task_to_keys[name_or_dataset]
        self.shuffled = shuffle
        if shuffle:
            self.dataset.shuffle()

    def _format_examples(self, examples):
        """
        Only for some task which has ONE input column, such as SST-2 and IMDB, NOT work for NLI such as MRPC.
        """
        text = [examples[self.input_columns[0]]]
        sentence = "".join(text)
        inputs = self.tokenizer(sentence, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt')
        output = int(examples['label'])
        return (inputs, output)

    def shuffle(self):
        self.dataset.shuffle()
        self.shuffled = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_examples(self.dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [
                self._format_examples(self.dataset[j]) for j in range(i.start, i.stop)
            ]


def get_select_metric(select_metric):
    if select_metric=="non_robust_perturbed_loss_mean+perturbed_loss_std_r":
        return "non_ro_p_loss_m+p_loss_s_r"
    elif select_metric=="non_robust_perturbed_loss_mean_r+perturbed_loss_std_r":
        return "non_ro_p_loss_mr+p_loss_s_r"
    elif select_metric=="perturbed_loss_mean_r+perturbed_loss_std_r":
        return "p_loss_mr+p_loss_s_r"
    elif select_metric=="perturbed_loss_mean+perturbed_loss_std_r":
        return "p_loss_m+p_loss_s_r"
    elif select_metric=="perturbed_loss_mean_r+perturbed_loss_std":
        return "p_loss_mr+p_loss_s"
    elif select_metric=="perturbed_loss_mean+perturbed_loss_std":
        return "p_loss_m+p_loss_s"
    return select_metric

