# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : speicial_attack.py
@Project  : Robust_Data
@Time     : 2022/10/1 15:06
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
import statistic_utils as statistic_utils

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='/root/Robust_Data_Outputs/retrained_models/glue_sst2_cycle_train4/from_bert-base-uncased_seed42_len67349_asp8_alr0.08_ep10_lr2e-05_interval1_with_ut1_use_cur0_to_bert-base-uncased_seed41_bsz32_lr2e-5_epochs16_select_metricloss_diff_mean_r_ratio0.2_prop0.5_two_end0_balan1_pgd8_0.06_oop1/epoch1')
    parser.add_argument('--model_type', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default="sst2", type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/Robust_Data/baselines/fine_tune/saved_models'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_lower_case', type=bool, default=True)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    # hyper-parameters
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--bias_correction', default=True)
    # parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir',type=str,default='/root/Robust_Data_Outputs/baselines/fine_tune/saved_models')
    # others
    parser.add_argument('--attack_every_epoch', default=False)
    parser.add_argument('-f', '--not_force_overwrite',action="store_true") # 只有传入了这个参数才会是true
    parser.add_argument('--cal_time',action="store_true")


    # select data
    parser.add_argument('--statistics_source', type=str, default="/root/Robust_Data/robust_statistics_datasetglue_tasksst2_lenNone_adv_steps10_adv_lr0.08_epoch5_lr2e-05_interval400.npy")
    parser.add_argument('--select_metric', type=str, default="loss_diff_mean")
    parser.add_argument('--select_metric2', type=str, default="loss_diff_mean_r")
    parser.add_argument('--select_ratio', type=float, default=0.3)
    parser.add_argument('--select_ratio2', type=float, default=0.3)
    parser.add_argument('--select_from_two_end', type=int, default=0)
    parser.add_argument('--ratio_proportion', type=float, default=0.5)
    parser.add_argument('--do_balance_labels', type=int, default=1)
    parser.add_argument('--with_untrained_model', default=1, type=int, help='')
    parser.add_argument('--use_cur_preds', default=0, type=int, help='whether use cur predictions or golden labels to calculate loss')
    parser.add_argument('--only_original_pred', default=1, type=int, help='是否只有预测正确了才会被纳入统计')
    parser.add_argument('--cycle_train', default=0, type=int, help='交替训练，如果大于0，则代表一次连续对某个subset训练多少epoch')
    parser.add_argument('--save_steps', default=-1, type=float, help="保存模型的step")
    parser.add_argument('--show_data', default=-1, type=int, help="展示选择的data")


    # few-shot setting
    parser.add_argument('--random_select', default=False, type=bool, help='')
    # parser.add_argument('--num_train_examples_per_class', default=-1, type=int, help='')
    parser.add_argument('--num_train_examples_ratio', default=1.0, type=float, help='')
    parser.add_argument('--data_indices_file', default=None, type=str, help='')

    # attack
    parser.add_argument('--do_attack', action="store_true")
    parser.add_argument('--attack_all', action="store_true")
    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.15, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)
    parser.add_argument("--results_file", default='attack_log.csv', type=str)
    parser.add_argument("--num_examples", default=1000, type=int)
    parser.add_argument("--attack_method", default="textfooler", type=str)
    # pgd attack
    parser.add_argument("--do_pgd_attack", default=1, type=int)
    parser.add_argument('--pgd_step', type=int, default=8)
    parser.add_argument('--pgd_lr', type=float, default=0.08)


    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    args = parser.parse_args()
    # if args.balance_labels:
    #     print("s")
    print(args.__dict__)

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

def do_pgd_attack(loader,
                  device, model,
                  adv_steps, adv_lr, adv_init_mag,
                  adv_norm_type,
                  adv_max_norm
                  ):
    model.train()
    pbar = tqdm(loader)
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)
        # initialize delta
        if adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)
            if adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embedding_init.size(-1)
                magnitude = adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1))
            elif adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-adv_init_mag,
                                                                  adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding_init)
        total_loss = 0.0
        for astep in range(adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            # logits = model(**batch).logits # todo ?不确定
            logits = model(**batch, return_dict=False)[0]  # todo ?不确定
            # _, preds = logits.max(dim=-1)
            # (1) backward
            losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            # loss = loss / adv_steps
            total_loss += loss.item()
            loss.backward()
            # loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()
                if adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > adv_max_norm).to(embedding_init)
                    reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()

            model.zero_grad()
            # optimizer.zero_grad()
            embedding_init = word_embedding_layer(input_ids)
        delta.requires_grad = False
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        model.zero_grad()
        # optimizer.zero_grad()
        logits = model(**batch).logits
        losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        batch_loss = loss.item()
        avg_loss.update(batch_loss)

        _, preds = logits.max(dim=-1)
        correct += (preds == labels.squeeze(-1)).sum().item()
        total += labels.size(0)
    pgd_accuracy = correct / (total + 1e-13)
    pgd_aua = pgd_accuracy
    logger.info(f'PGD Aua: {pgd_accuracy}')
    logger.info(f'PGD Loss: {avg_loss}')
    return pgd_accuracy




def special_attack(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels,mirror='tuna')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    if args.model_type =="bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(args.model_name,config=config)

    elif args.model_type == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_name,config=config)

    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news' or args.dataset_name=="SetFit/20_newsgroups":
        args.task_name = None
        args.valid = "test"

    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)

    from finetune_with_select_data import generate_data_indices
    if args.statistics_source != None:
        result_data_indices, selected_label_nums = generate_data_indices(args, train_dataset,args.select_metric,args.select_ratio)
        train_dataset.dataset = train_dataset.dataset.select(result_data_indices)

        if args.cycle_train>0:
            train_dataset2 = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                      subset=args.task_name)
            result_data_indices2, selected_label_nums2 = generate_data_indices(args, train_dataset2, args.select_metric2,
                                                                             args.select_ratio2)

            train_dataset2.dataset = train_dataset2.dataset.select(result_data_indices2)
            # train_loader2 = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    args.selected_label_nums = str(selected_label_nums)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    logger.info("train dataset length: "+ str(len(train_dataset)))
    # for dev
    dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split=args.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # do pgd attack
    do_pgd_attack(train_loader,device,model,
                  args.pgd_step,args.pgd_lr,args.adv_init_mag,args.adv_norm_type,args.adv_max_norm)


if __name__ == '__main__':
    args = parse_args()

    level = logging.INFO
    logging.basicConfig(level=level)

    special_attack(args)