# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : Data_Statistics.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 18:56
@Author   : Zhiheng Xi
"""

import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)
from models.modeliing_bert import BertForSequenceClassification
from models.modeling_roberta import RobertaForSequenceClassification
from torch import nn
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import utils as utils
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
# from modeling_utils import PreTrainedModel
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default="sst2", type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/Robust_Data/baselines/fine_tune/saved_models'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # hyper-parameters
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    # parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir',type=str,default='/root/tmp_dir')
    # others
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--attack_every_epoch', default=False)
    parser.add_argument('-f', '--not_force_overwrite',action="store_true") # 只有传入了这个参数才会是true
    parser.add_argument('--cal_time',action="store_true")

    # few-shot setting
    parser.add_argument('--random_select', default=False, type=bool, help='')
    # parser.add_argument('--num_train_examples_per_class', default=-1, type=int, help='')
    parser.add_argument('--num_train_examples_ratio', default=1.0, type=float, help='')
    parser.add_argument('--data_indices_file', default=None, type=str, help='')

    # Adversarial training specific
    parser.add_argument('--adv_steps', default=5, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_lr', default=0.03, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')
    parser.add_argument('--use_fgsm', default=0, type=float, help='')


    # robust data statistics
    parser.add_argument('--statistic_interval', default=1, type=float, help='')
    parser.add_argument('--dataset_len', default=None, type=int, help='')
    parser.add_argument('--with_untrained_model', default=1, type=int, help='')
    parser.add_argument('--use_cur_preds', default=0, type=int, help='whether use cur predictions or golden labels to calculate loss')
    parser.add_argument('--do_train_shuffle', default=1, type=int, help='')

    args = parser.parse_args()

    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def robust_statistics(model,train_dev_loader,train_set_len,device,use_cur_preds=True):
    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics={}
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["golden_label"] = cur_label.item()
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            statistics[data_index]["original_logit"] = cur_logits[cur_label.item()].item()
            statistics[data_index]["original_probability"] = nn.Softmax(dim=-1)(cur_logits)[cur_label.item()].item()

            data_index+=1
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        # for freelb
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        if args.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)
            if args.adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif args.adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                                  args.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)
        total_loss = 0.0
        for astep in range(args.adv_steps):
            # 0. forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = model(**batch, return_dict=False)[0]
            _, preds = logits.max(dim=-1)
            # 1.
            if use_cur_preds:
                losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
            else:
                losses = F.cross_entropy(logits, labels.squeeze(-1))
            # losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            loss = loss / args.adv_steps
            total_loss += loss.item()
            loss.backward()

            if astep == args.adv_steps - 1:

                for i in range(len(labels)):
                    cur_logits = logits[i]
                    cur_label = labels[i]
                    cur_pred = preds[i]
                    if use_cur_preds:
                        cur_batch_pred = cur_batch_preds[i]
                        cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
                    else:
                        cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
                    cur_loss = torch.mean(cur_losses)
                    statistics[data_index]["after_perturb_loss"] = cur_loss.item()
                    statistics[data_index]["after_perturb_pred"] = (cur_label.item() == cur_pred.item())

                    statistics[data_index]["after_perturb_logit"] = cur_logits[cur_label.item()].item()
                    statistics[data_index]["after_perturb_probability"] = nn.Softmax(dim=-1)(cur_logits)[
                        cur_label.item()].item()

                    statistics[data_index]["logit_diff"] = statistics[data_index]["after_perturb_logit"] - statistics[data_index]["original_logit"]
                    statistics[data_index]["probability_diff"] = statistics[data_index]["after_perturb_probability"] - statistics[data_index]["original_probability"]

                    statistics[data_index]["loss_diff"] = statistics[data_index]["after_perturb_loss"] - statistics[data_index]["original_loss"]
                    statistics[data_index]["normed_loss_diff"] = statistics[data_index]["loss_diff"] / delta.norm(p=2,dim=(1,2),keepdim=False)[i].item()
                    data_index += 1
                break


            # 2. get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # 3. update and clip
            if args.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + args.adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
                if args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                    reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                        1)
                    delta = (delta * reweights).detach()
            elif args.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                         1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + args.adv_lr * delta_grad / denorm).detach()

            embedding_init = word_embedding_layer(input_ids)  # 重新初始化embedding
        pbar.set_description("Doing perturbation statistics")

    return statistics


def robust_statistics_fgsm(model,train_dev_loader,train_set_len,device,use_cur_preds=True):

    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics={}
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            data_index+=1
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        # for freelb
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0

        # 0. forward
        embedding_init.requires_grad_()
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        total_loss += loss.item()
        loss.backward()
        # 2. get gradient on delta
        delta_grad = delta.grad.clone().detach()
        # 3. update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                    1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                     1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()


        # 4. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        # if use_cur_preds:
        #     losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        # else:
        #     losses = F.cross_entropy(logits, labels.squeeze(-1))
        # # losses = F.cross_entropy(logits, labels.squeeze(-1))
        # loss = torch.mean(losses)
        # loss = loss / args.adv_steps
        # total_loss += loss.item()
        # loss.backward()

        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_batch_pred = cur_batch_preds[i]
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_batch_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["after_perturb_loss"] = cur_loss.item()
            statistics[data_index]["after_perturb_pred"] = (cur_label.item() == cur_pred.item())
            statistics[data_index]["loss_diff"] = statistics[data_index]["after_perturb_loss"] - statistics[data_index][
                "original_loss"]
            statistics[data_index]["normed_loss_diff"] = statistics[data_index]["loss_diff"] / \
                                                         delta.norm(p=2, dim=(1, 2), keepdim=False)[i].item()
            statistics[data_index]["delta_grad"] = delta_grad.norm(p=2, dim=(1, 2), keepdim=False)[i].item()
            data_index += 1

        pbar.set_description("Doing perturbation statistics")


    return statistics

def robust_statistics_grad(model,train_dev_loader,train_set_len,device,use_cur_preds=True):
    """
    Collect statistics
    :param model:
    :param train_dev_loader:
    :param train_set_len:
    :param device:
    :param use_cur_preds:
    :return:
    """
    pbar = tqdm(train_dev_loader)
    model.eval()
    statistics={}
    for i in range(train_set_len):
        statistics[i] = {}

    data_index = 0
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        for i in range(len(labels)):
            cur_logits = logits[i]
            cur_label = labels[i]
            cur_pred = preds[i]
            if use_cur_preds:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0), cur_pred.unsqueeze(0))
            else:
                cur_losses = F.cross_entropy(cur_logits.unsqueeze(0),cur_label.unsqueeze(0))
            cur_loss = torch.mean(cur_losses)
            statistics[data_index]["original_loss"] = cur_loss.item()
            statistics[data_index]["original_pred"] = (cur_label.item()==cur_pred.item())
            data_index+=1
        pbar.set_description("Doing original statistics")
        # pass

    data_index = 0
    model.train()
    pbar = tqdm(train_dev_loader)
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        if use_cur_preds:

            cur_batch_logits = model(**model_inputs).logits
            _, cur_batch_preds = cur_batch_logits.max(dim=-1)

        model.zero_grad()
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0

        # 0. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        if use_cur_preds:
            losses = F.cross_entropy(logits, cur_batch_preds.squeeze(-1))
        else:
            losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        total_loss += loss.item()
        loss.backward()
        # 2. get gradient on delta


    model.zero_grad()

    return statistics



def finetune(args):
    set_seed(args.seed)
    # output_dir = Path(args.output_dir)
    # if not output_dir.exists():
    #     logger.info(f'Making checkpoint directory: {output_dir}')
    #     output_dir.mkdir(parents=True)
    # elif args.not_force_overwrite:
    #     return
    # log_file = os.path.join(output_dir, 'INFO.log')

    if args.dataset_name == "imdb":
        num_labels = 2
        args.num_labels = 2
        output_mode = "classification"
    elif args.dataset_name == "ag_news":
        num_labels = 4
        args.num_labels = 4
        output_mode = "classification"

    # pre-trained config/tokenizer/model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels,mirror='tuna')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name =="bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(args.model_name,config=config)

    elif args.model_name == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_name,config=config)

    model.to(device)

    # prepare datasets
    # logger.info(utils.say())
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news' or args.dataset_name=="SetFit/20_newsgroups":
        args.task_name = None
        args.valid = "test"

    elif args.task_name=="mnli":
        args.valid = "validation_matched"
        output_mode = "classification"

    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    if args.dataset_len is not None and args.dataset_len < len(train_dataset):
        # total_size =
        train_dataset.dataset = train_dataset.dataset.train_test_split(1,args.dataset_len,seed=42)["train"]
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=args.do_train_shuffle, collate_fn=collator)

    train_dev_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)

    train_dev_dict = {}
    logger.info("train dataset length: "+ str(len(train_dataset)))
    # for dev
    dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split=args.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # for test
    if args.do_test:
        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=args.adam_epsilon,
                      correct_bias=args.bias_correction
    )


    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    epoch_steps = len(train_dataset) // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    robust_statistics_dict = {}

    # num_record_steps = num_training_steps//args.statistic_interval + 1
    # robust_statistics_records = [[] for _ in range(num_record_steps)]

    try:
        import time
        best_accuracy = 0
        global_step = 0
        for epoch in range(args.epochs):
            avg_loss = utils.ExponentialMovingAverage()
            model.train()
            pbar = tqdm(train_loader)
            for model_inputs, labels in pbar:
                if args.with_untrained_model:
                    if global_step % int(args.statistic_interval * epoch_steps) == 0:
                            # and global_step!=0\

                        if args.use_fgsm:
                            cur_robust_statistics = robust_statistics_fgsm(model, train_dev_loader,
                                                                      train_set_len=len(train_dataset), device=device,
                                                                      use_cur_preds=args.use_cur_preds)
                        else:
                            cur_robust_statistics = robust_statistics(model,train_dev_loader,train_set_len=len(train_dataset),device=device,use_cur_preds=args.use_cur_preds)

                        robust_statistics_dict[global_step] = cur_robust_statistics
                else:
                    if global_step % int(args.statistic_interval * epoch_steps) == 0 and global_step!=0:
                        if args.use_fgsm:
                            cur_robust_statistics = robust_statistics_fgsm(model, train_dev_loader,
                                                                           train_set_len=len(train_dataset),
                                                                           device=device,
                                                                           use_cur_preds=args.use_cur_preds)
                        else:
                            cur_robust_statistics = robust_statistics(model, train_dev_loader,
                                                                      train_set_len=len(train_dataset), device=device,
                                                                      use_cur_preds=args.use_cur_preds)
                        robust_statistics_dict[global_step] = cur_robust_statistics

                batch_loss=0
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                model.zero_grad()
                logits = model(**model_inputs,return_dict=False)[0]
                _, preds = logits.max(dim=-1)

                losses = F.cross_entropy(logits,labels.squeeze(-1))
                loss = torch.mean(losses)

                # loss2  = model(**model_inputs,return_dict=False)
                batch_loss=loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                avg_loss.update(batch_loss)

                pbar.set_description(f'epoch: {epoch: d}, '
                                     f'loss: {avg_loss.get_metric(): 0.4f}, '
                                     f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
                global_step+=1
            # s = Path(str(output_dir) + '/epoch' + str(epoch))
            # if not s.exists():
            #     s.mkdir(parents=True)
            # model.save_pretrained(s)
            # tokenizer.save_pretrained(s)
            # torch.save(args, os.path.join(s, "training_args.bin"))

            if args.do_eval and not args.cal_time:
                logger.info('Evaluating...')
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for model_inputs, labels in dev_loader:
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        labels = labels.to(device)
                        logits = model(**model_inputs,return_dict=False)[0]
                        _, preds = logits.max(dim=-1)
                        correct += (preds == labels.squeeze(-1)).sum().item()
                        total += labels.size(0)
                    accuracy = correct / (total + 1e-13)
                logger.info(f'Epoch: {epoch}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Accuracy: {accuracy}')

                if accuracy > best_accuracy:
                    logger.info('Best performance so far.')
                    # model.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch
            # logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')
        # save statistics


        if args.output_dir=='/root/tmp_dir':
            if args.use_fgsm:
                np.save(
                    'fgsm/robust_statistics_model{}_dataset{}_task{}_seed{}_shuffle{}_len{}_adv_steps{}_adv_lr{}_epoch{}_lr{}_interval{}_with_untrained_model{}_use_cur_preds{}.npy'
                    .format(args.model_name, args.dataset_name, args.task_name, args.seed, args.do_train_shuffle,
                            args.dataset_len,
                            args.adv_steps, args.adv_lr, args.epochs, args.lr,
                            args.statistic_interval, args.with_untrained_model, args.use_cur_preds
                            ),
                    robust_statistics_dict)
            else:
                np.save('robust_statistics_model{}_dataset{}_task{}_seed{}_shuffle{}_len{}_adv_steps{}_adv_lr{}_epoch{}_lr{}_interval{}_with_untrained_model{}_use_cur_preds{}.npy'
                        .format(args.model_name,args.dataset_name,args.task_name,args.seed,args.do_train_shuffle,
                                args.dataset_len,
                                args.adv_steps,args.adv_lr,args.epochs,args.lr,
                                args.statistic_interval,args.with_untrained_model,args.use_cur_preds
                                ),
                        robust_statistics_dict)
        else:
            np.save(args.output_dir,robust_statistics_dict)
    except KeyboardInterrupt:
        logger.info('Interrupted...')

if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    finetune(args)
