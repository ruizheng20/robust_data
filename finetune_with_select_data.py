# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : finetune_with_select_data.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 18:59
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
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default="sst2", type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/Robust_Data/baselines/fine_tune/saved_models'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_eval', type=int, default=1)
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
    parser.add_argument('--output_dir',type=str,default='/root/Robust_Data_Outputs/baselines/fine_tune/saved_models')
    # others
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--attack_every_epoch', default=0,type=int,help="10:攻击train dataset1；20：攻击 train dataset2；否则攻击dev dataset")
    parser.add_argument('--attack_every_step', default=0,type=int)
    parser.add_argument('-f', '--not_force_overwrite',action="store_true") # 只有传入了这个参数才会是true
    parser.add_argument('--cal_time',action="store_true")

    # few-shot setting
    parser.add_argument('--random_select', default=False, type=bool, help='')
    # parser.add_argument('--num_train_examples_per_class', default=-1, type=int, help='')
    parser.add_argument('--num_train_examples_ratio', default=1.0, type=float, help='')
    parser.add_argument('--data_indices_file', default=None, type=str, help='')

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
    parser.add_argument('--reinit_classifier', default=0, type=int, help="")
    parser.add_argument('--freeze_bert', default=0, type=int, help="")

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
    parser.add_argument("--do_pgd_attack", default=0, type=int)
    parser.add_argument('--pgd_step', type=int, default=5)
    parser.add_argument('--pgd_lr', type=float, default=0.05)
    # freelb
    parser.add_argument('--freelb_adv_steps', type=int, default=5)
    parser.add_argument('--freelb_adv_lr', type=float, default=0.03)
    parser.add_argument('--do_freelb_training', type=int, default=0)

    # parser.add_argument('--adv_steps', default=5, type=int,
    #                     help='Number of gradient ascent steps for the adversary')
    # parser.add_argument('--adv_lr', default=0.03, type=float,
    #                     help='Step size of gradient ascent')
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
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
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

def finetune(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif args.not_force_overwrite:
        return
    log_file = os.path.join(output_dir, 'INFO.log')

    if args.dataset_name == "imdb":
        num_labels = 2
        output_mode = "classification"
    elif args.dataset_name == "ag_news":
        num_labels = 4
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
    if args.reinit_classifier:
        model.reinit_classifier()
    if args.freeze_bert:
        model.freeze_Bert()

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

    if args.statistics_source != None:
        result_data_indices, selected_label_nums = generate_data_indices(args, train_dataset,args.select_metric,args.select_ratio)
        train_dataset.dataset = train_dataset.dataset.select(result_data_indices)
        # train_dataset.dataset = train_dataset.dataset.select(result_data_indices[0:5])

        if args.cycle_train>0:
            train_dataset2 = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                      subset=args.task_name)
            result_data_indices2, selected_label_nums2 = generate_data_indices(args, train_dataset2, args.select_metric2,
                                                                             args.select_ratio2)

            train_dataset2.dataset = train_dataset2.dataset.select(result_data_indices2)
            # train_loader2 = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.show_data > 0:
        import csv
        print("Metric:{}".format(args.select_metric))
        if args.dataset_name=="glue" and args.task_name=="qqp":
            raw_data_sentence1 = "question1"
            raw_data_sentence2 = "question2"
            for i in range(args.show_data):
                # print("sentence: {}, label: {}".format(train_dataset.dataset[raw_data_sentence][i],train_dataset.dataset["label"][i]))
                # print()
                # print()
                show_data_dir = "/root/Robust_Data/analysis_experiments/show_data/{}_{}/".format(args.dataset_name,
                                                                                                 args.task_name)
                if not os.path.exists(show_data_dir):
                    os.makedirs(show_data_dir)
                show_data_file = show_data_dir + "show_data{}.csv".format(args.select_metric)
                show_data_format = [
                    "select_metric", "seed", "question1","question2", "label", "order_in_cur_metric"
                ]
                if not os.path.exists(show_data_file):
                    # os.makedirs(show_data_dir)
                    out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
                    csv_writer = csv.writer(out_csv)
                    cur_row = [i for i in show_data_format]
                    csv_writer.writerow(cur_row)
                    out_csv.close()

                # 写入数据
                out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
                csv_writer = csv.writer(out_csv)
                cur_row = []
                cur_row.append(args.select_metric)
                cur_row.append(args.seed)
                cur_row.append(train_dataset.dataset["question1"][i])
                cur_row.append(train_dataset.dataset["question2"][i])
                cur_row.append(train_dataset.dataset["label"][i])
                cur_row.append(i)
                csv_writer.writerow(cur_row)
                out_csv.close()

        else:
            if args.dataset_name=="glue" and args.task_name=="sst2":
                raw_data_sentence = "sentence"

            elif args.dataset_name=="imdb" :
                raw_data_sentence = "text"
            else:
                raw_data_sentence = "text"
            for i in range(args.show_data):
                # print("sentence: {}, label: {}".format(train_dataset.dataset[raw_data_sentence][i],train_dataset.dataset["label"][i]))
                # print()
                # print()
                show_data_dir = "/root/Robust_Data/analysis_experiments/show_data/{}_{}/".format(args.dataset_name,args.task_name)
                if not os.path.exists(show_data_dir):
                    os.makedirs(show_data_dir)
                show_data_file = show_data_dir+"show_data{}.csv".format(args.select_metric)
                show_data_format = [
                   "select_metric","seed", "sentence","label","order_in_cur_metric"
                ]
                if not os.path.exists(show_data_file):
                    # os.makedirs(show_data_dir)
                    out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
                    csv_writer = csv.writer(out_csv)
                    cur_row = [i for i in show_data_format]
                    csv_writer.writerow(cur_row)
                    out_csv.close()

                # 写入数据
                out_csv = open(show_data_file, 'a', encoding='utf-8', newline="")
                csv_writer = csv.writer(out_csv)
                cur_row = []
                cur_row.append(args.select_metric)
                cur_row.append(args.seed)
                cur_row.append(train_dataset.dataset[raw_data_sentence][i])
                cur_row.append(train_dataset.dataset["label"][i])
                cur_row.append(i)
                csv_writer.writerow(cur_row)
                out_csv.close()

        return

    print(str(selected_label_nums))
    args.selected_label_nums = str(selected_label_nums)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
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
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    one_epoch_steps = int(len(train_dataset) // args.bsz)
    if args.save_steps<1 and args.save_steps>0:
        args.save_steps = int(one_epoch_steps*args.save_steps)
    save_steps = args.save_steps
    try:
        best_accuracy = 0
        global_step = 0
        for epoch in range(args.epochs):
            avg_loss = utils.ExponentialMovingAverage()

            model.train()
            # train 1 epoch
            if args.cycle_train>0:
                if epoch%(args.cycle_train * 2) < args.cycle_train: #
                    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    logger.info("current metric:{},cureent ratio:{}".format(args.select_metric,args.select_ratio))

                else:
                    train_loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    logger.info("current metric:{},cureent ratio:{}".format(args.select_metric2,args.select_ratio2))

            if args.do_freelb_training > 0:
                global_step = freelb_one_epoch(args, avg_loss, device, epoch, model, optimizer, scheduler, train_loader,
                                              global_step, save_steps, tokenizer,
                                               args.adv_init_mag, args.adv_norm_type, args.freelb_adv_steps,
                                               args.freelb_adv_lr,
                                               args.adv_max_norm
                                               )
            else:
                global_step = train_one_epoch(args,avg_loss, device, epoch, model, optimizer, scheduler, train_loader,global_step,save_steps,tokenizer)
            # save model
            save_model_one_epoch(args, epoch, model, output_dir, tokenizer)
            # eval model
            if args.do_eval:
                accuracy,clean_loss = evaluate(dev_loader, device, model)
                logger.info(f'Epoch: {epoch}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Accuracy: {accuracy}')
                if accuracy > best_accuracy:
                    logger.info('Best performance so far.')
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    best_accuracy = accuracy
                    best_dev_epoch = epoch
        if args.attack_every_step > 0:
            for step in range(num_training_steps):
                if step%args.save_steps==0:

                    one_epoch_steps = int(len(train_dataset) // args.bsz)

                    epoch = int(step//one_epoch_steps)
                    logger.info("current step:{},current epoch:{}".format(step, epoch))
                    if epoch%(args.cycle_train * 2) < args.cycle_train: #
                        logger.info("current metric:{},cureent ratio:{}".format(args.select_metric,args.select_ratio))
                        cur_select_metric = args.select_metric
                        args.cur_select_metric = cur_select_metric

                    else:
                        logger.info("current metric:{},cureent ratio:{}".format(args.select_metric2,args.select_ratio2))
                        cur_select_metric = args.select_metric2
                        args.cur_select_metric = cur_select_metric
                    args.cur_epoch = epoch
                    args.cur_step = step
                    s = Path(str(output_dir) + '/step' + str(step))
                    if args.model_name == "bert-base-uncased":
                        model = BertForSequenceClassification.from_pretrained(s, config=config)
                    elif args.model_name == "roberta-base":
                        model = RobertaForSequenceClassification.from_pretrained(s, config=config)
                    else:
                        model = BertForSequenceClassification.from_pretrained(s, config=config)
                    model.to(device)
                    if args.do_eval and not args.cal_time:
                        clean,clean_loss = evaluate(dev_loader, device, model)
                        args.clean = clean
                        args.clean_loss = clean_loss.get_metric()
                    else:
                        args.clean=0
                        args.clean_loss=0
                    if args.do_pgd_attack:
                        pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                                 adv_steps=args.pgd_step,
                                                 adv_lr=args.pgd_lr,
                                                 adv_init_mag=args.adv_init_mag,
                                                 adv_max_norm=args.adv_max_norm,
                                                 adv_norm_type=args.adv_norm_type
                                                 )
                        model.train()
                        optimizer.zero_grad()

                        args.pgd_aua = pgd_aua
                        args.pgd_loss = pgd_loss.get_metric()
                    else:
                        args.pgd_aua = 0
                        args.pgd_loss = 0


                    do_textattack_attack(args, model, tokenizer,
                                         do_attack=args.do_attack,
                                         attack_seed=42,
                                         attack_all=args.attack_all,
                                         attack_method="textfooler",
                                         attack_every_epoch=False,
                                         attack_every_step=True
                                         )
                    model.train()

        elif args.attack_every_epoch>0:
            for epoch in range(args.epochs):
                logger.info("current epoch:{}".format( epoch))
                if args.cycle_train > 0:
                    if epoch%(args.cycle_train * 2) < args.cycle_train: #
                        logger.info("current metric:{},cureent ratio:{}".format(args.select_metric,args.select_ratio))
                        cur_select_metric = args.select_metric
                        args.cur_select_metric = cur_select_metric

                    else:
                        logger.info("current metric:{},cureent ratio:{}".format(args.select_metric2,args.select_ratio2))
                        cur_select_metric = args.select_metric2
                        args.cur_select_metric = cur_select_metric
                else:
                    logger.info("current metric:{},cureent ratio:{}".format(args.select_metric, args.select_ratio))
                    cur_select_metric = args.select_metric
                    args.cur_select_metric = cur_select_metric

                args.cur_epoch = epoch
                s = Path(str(output_dir) + '/epoch' + str(epoch))
                if args.model_name == "bert-base-uncased":
                    model = BertForSequenceClassification.from_pretrained(s, config=config)
                elif args.model_name == "roberta-base":
                    model = RobertaForSequenceClassification.from_pretrained(s, config=config)
                else:
                    model = BertForSequenceClassification.from_pretrained(s, config=config)
                model.to(device)
                if args.do_eval and not args.cal_time:
                    if args.attack_every_epoch ==10:
                        args.attack_dataset_metric = args.select_metric
                        loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    elif args.attack_every_epoch ==20:
                        args.attack_dataset_metric = args.select_metric2
                        loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    else:
                        args.attack_dataset_metric = "test_set"
                        loader = dev_loader
                    clean,clean_loss = evaluate(loader, device, model)
                    args.clean = clean
                    args.clean_loss = clean_loss.get_metric()
                else:
                    args.clean = 0
                    args.clean_loss = 0
                if args.do_pgd_attack:
                    if args.attack_every_epoch ==10:
                        args.attack_dataset_metric = args.select_metric
                        loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    elif args.attack_every_epoch ==20:
                        args.attack_dataset_metric = args.select_metric2
                        loader = DataLoader(train_dataset2, batch_size=args.bsz, shuffle=True, collate_fn=collator)
                    else:
                        args.attack_dataset_metric = "test_set"
                        loader = dev_loader
                    pgd_aua,pgd_loss = do_pgd_attack( loader, device, model,
                                             adv_steps=args.pgd_step,
                                             adv_lr=args.pgd_lr,
                                             adv_init_mag=args.adv_init_mag,
                                             adv_max_norm=args.adv_max_norm,
                                             adv_norm_type=args.adv_norm_type
                                             )
                    optimizer.zero_grad()
                    args.pgd_aua = pgd_aua
                    args.pgd_loss = pgd_loss.get_metric()
                else:
                    args.pgd_aua = 0
                    args.pgd_loss = 0

                do_textattack_attack(args, model, tokenizer,
                                     do_attack=args.do_attack,
                                     attack_seed=42,
                                     attack_all=args.attack_all,
                                     attack_method="textfooler",
                                     attack_every_epoch=True,
                                     attack_every_step=False
                                     )
                model.train()
        if args.do_pgd_attack:
            pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                             adv_steps=args.pgd_step,
                                             adv_lr=args.pgd_lr,
                                             adv_init_mag=args.adv_init_mag,
                                             adv_max_norm=args.adv_max_norm,
                                             adv_norm_type=args.adv_norm_type
                                             )
            optimizer.zero_grad()
            args.pgd_aua = pgd_aua
        else:
            args.pgd_aua = 0
            args.pgd_loss = 0
        if not args.attack_every_epoch and not args.attack_every_step and args.do_attack:
            if args.do_eval and not args.cal_time:
                args.attack_dataset_metric = "test_set"
                loader = dev_loader
                clean, clean_loss = evaluate(loader, device, model)
                args.clean = clean
                args.clean_loss = clean_loss
            else:
                args.clean=0
                args.clean_loss=0
            do_textattack_attack(args, model, tokenizer,
                                 do_attack=args.do_attack,
                                 attack_seed=42,
                                 attack_all=args.attack_all,
                                 attack_method="textfooler",
                                 attack_every_epoch=args.attack_every_epoch
                                 )

    except KeyboardInterrupt:
        logger.info('Interrupted...')


def save_model_one_epoch(args, epoch, model, output_dir, tokenizer):
    s = Path(str(output_dir) + '/epoch' + str(epoch))
    if not s.exists():
        s.mkdir(parents=True)
    model.save_pretrained(s)
    tokenizer.save_pretrained(s)
    torch.save(args, os.path.join(s, "training_args.bin"))

def save_model_one_step(args, global_step, model, output_dir, tokenizer):
    logger.info("save model at step {}".format(global_step))
    s = Path(str(output_dir) + '/step' + str(global_step))
    if not s.exists():
        s.mkdir(parents=True)
    model.save_pretrained(s)
    tokenizer.save_pretrained(s)
    torch.save(args, os.path.join(s, "training_args.bin"))

def freelb_one_epoch(args,avg_loss, device, epoch, model, optimizer, scheduler,
                     train_loader,global_step,save_steps,tokenizer,
                     adv_init_mag,adv_norm_type,adv_steps,adv_lr,
                        adv_max_norm
                     ):
    pbar = tqdm(train_loader)
    logger.info('Freelb training epoch {}...'.format(epoch))

    for model_inputs, labels in pbar:
        if save_steps > 0 and global_step % save_steps ==0:
            save_model_one_step(args,global_step,model,output_dir=args.output_dir,tokenizer=tokenizer)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()

        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)

        # initialize delta
        if adv_init_mag > 0:
            input_mask = attention_mask.to(embedding_init)  # 对embedding做mask？
            input_lengths = torch.sum(input_mask, 1)
            if adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(
                    2)
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
            # 0. forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = model(**batch, return_dict=False)[0]
            _, preds = logits.max(dim=-1)
            # 1. backwatd
            losses = F.cross_entropy(logits, labels)
            loss = torch.mean(losses)
            loss = loss / adv_steps
            total_loss += loss.item()
            loss.backward()
            if astep == adv_steps - 1:
                break
            # 2. get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # 3. update and clip
            if adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
                if adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > adv_max_norm).to(embedding_init)
                    reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                        1)
                    delta = (delta * reweights).detach()
            elif adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1,
                                                                                                         1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + adv_lr * delta_grad / denorm).detach()

            embedding_init = word_embedding_layer(input_ids)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        avg_loss.update(total_loss)
        pbar.set_description(f'epoch: {epoch: d}, '
                             f'loss: {avg_loss.get_metric(): 0.4f}, '
                             f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
        global_step += 1

    return global_step

def train_one_epoch(args,avg_loss, device, epoch, model, optimizer, scheduler, train_loader,global_step,save_steps,tokenizer):
    pbar = tqdm(train_loader)
    for model_inputs, labels in pbar:
        if save_steps > 0 and global_step % save_steps ==0:
            save_model_one_step(args,global_step,model,output_dir=args.output_dir,tokenizer=tokenizer)
        batch_loss = 0
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        logits = model(**model_inputs, return_dict=False)[0]
        _, preds = logits.max(dim=-1)

        losses = F.cross_entropy(logits, labels)
        loss = torch.mean(losses)

        batch_loss = loss.item()
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

    return global_step


def generate_data_indices(args, train_dataset,select_metric,select_ratio):
    if args.select_ratio==1.0 or (args.select_ratio<0.99 and args.select_ratio > 1.01):
        result_data_indices = list(range(len(train_dataset)))
        selected_label_nums = "None"
        return result_data_indices,selected_label_nums

    len_dataset, _ = utils.dataset_to_length_and_batch_size(args.dataset_name, args.task_name)
    new_data_loss_diff, new_data_original_correctness, new_data_flip_times, \
    new_data_delta_grad, new_data_original_loss, new_data_perturbed_loss, new_data_original_logit, \
    new_data_perturbed_logit, new_data_logit_diff, new_data_original_probability, \
    new_data_perturbed_probability, new_data_probability_diff,new_data_golden_label = statistic_utils.process_npy(
                                                                                args.statistics_source
                                                                                , len_dataset, only_original_pred=args.only_original_pred)
    df = statistic_utils.data_with_metrics(new_data_loss_diff,
                                           new_data_original_correctness,
                                           new_data_flip_times,
                                           new_data_delta_grad,
                                           new_data_original_loss,
                                           new_data_perturbed_loss,
                                           new_data_original_logit=new_data_original_logit,
                                           new_data_perturbed_logit=new_data_perturbed_logit,
                                           new_data_logit_diff=new_data_logit_diff,
                                           new_data_original_probability=new_data_original_probability,
                                           new_data_perturbed_probability=new_data_perturbed_probability,
                                           new_data_probability_diff=new_data_probability_diff,
                                           new_data_golden_label=new_data_golden_label,
                                           do_norm=True # todo
                                           )
    if not args.select_from_two_end:
        selected_label_nums, result_data_indices = select_data(select_metric,
                                                               select_ratio,
                                                               args.num_labels,
                                                               args.do_balance_labels,
                                                               df, train_dataset)
    else:
        tmp_ratio = select_ratio
        tmp_select_metric = select_metric
        select_ratio = select_ratio * args.ratio_proportion

        selected_label_nums, result_data_indices = select_data(select_metric,
                                                               select_ratio,
                                                               args.num_labels,
                                                               args.do_balance_labels,
                                                               df, train_dataset)
        select_ratio = tmp_ratio
        select_ratio = select_ratio * (1 - args.ratio_proportion)
        select_metric = select_metric[:-2] if select_metric[-2:] == "_r" else select_metric + "_r"
        selected_label_nums_2, result_data_indices_2 = select_data(select_metric,
                                                                   select_ratio,
                                                                   args.num_labels,
                                                                   args.do_balance_labels,
                                                                   df, train_dataset)
        select_ratio = tmp_ratio
        select_metric = tmp_select_metric
        for i in selected_label_nums_2.keys():
            selected_label_nums[i] += selected_label_nums_2[i]
        result_data_indices = result_data_indices + result_data_indices_2
    return result_data_indices, selected_label_nums



def do_textattack_attack(args,model, tokenizer,
                         do_attack=False,attack_seed=42,attack_all=False,
                         attack_method="textfooler",attack_every_epoch=False,attack_every_step=False,log_format=None):
    # attack_seed = 42
    model.eval()
    from attack.attack_all import do_attack_with_model

    if attack_all:
        attack_methods = ["textfooler", "textbugger",
                          "bertattack"]
    else:
        attack_methods = [attack_method]
    # print(str(attack_all))
    # print(str(attack_methods))
    if log_format==None:
        log_format = ["statistics_source",
                       "select_metric",
                       "select_ratio", "ratio_proportion",
                       "selected_label_nums",
                       "lr",
                       "seed", "epochs",
                       "pgd_step", "pgd_lr",
                       "clean", "pgd_aua", "attack_method"
                     ]

    if attack_every_epoch:
        log_format.append("cur_epoch")
        log_format.append("cur_select_metric")
        log_format.append("cycle_train")
        log_format.append("clean_loss")
        log_format.append("pgd_loss")
        log_format.append("attack_every_epoch")
        log_format.append("attack_dataset_metric")
    elif attack_every_step:
        log_format.append("cur_epoch")
        log_format.append("save_steps")
        log_format.append("cur_step")
        log_format.append("cur_select_metric")
        log_format.append("cycle_train")
        log_format.append("clean_loss")
        log_format.append("pgd_loss")
        log_format.append("attack_every_step")
        log_format.append("attack_dataset_metric")

    for attack_method in attack_methods:
        args.attack_method = attack_method

        args_dict = args.__dict__
        data_row = [args_dict[i] for i in log_format]
        do_attack_with_model(model,tokenizer,
                             dataset_name=args.dataset_name,
                             task_name=args.task_name,valid=args.valid,
                             attack_method=args.attack_method,
                             num_examples=args.num_examples,attack_seed=42,
                             results_file=args.results_file,
                             seed=args.seed,
                             model_name=args.model_name,
                             log_format=log_format,
                             data_row=data_row
                             )
        # do_attack_with_model(args, model, tokenizer
        #                      , log_format=log_format)

    model.train()
    model.zero_grad()

def do_pgd_attack( dev_loader,
                  device, model,
                  adv_steps,adv_lr,adv_init_mag,
                  adv_norm_type,
                  adv_max_norm
                  ):
    model.eval()
    pbar = tqdm(dev_loader)
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
        # if adv_init_mag > 0:
        #     input_mask = attention_mask.to(embedding_init)
        #     input_lengths = torch.sum(input_mask, 1)
        #     if adv_norm_type == 'l2':
        #         delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
        #         dims = input_lengths * embedding_init.size(-1)
        #         magnitude = adv_init_mag / torch.sqrt(dims)
        #         delta = (delta * magnitude.view(-1, 1, 1))
        #     elif adv_norm_type == 'linf':
        #         delta = torch.zeros_like(embedding_init).uniform_(-adv_init_mag,
        #                                                           adv_init_mag) * input_mask.unsqueeze(2)
        # else:
        delta = torch.zeros_like(embedding_init)
        total_loss = 0.0
        for astep in range(adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            # logits = model(**batch).logits
            logits = model(**batch, return_dict=False)[0]
            # _, preds = logits.max(dim=-1)
            # (1) backward
            losses = F.cross_entropy(logits, labels)
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

        losses = F.cross_entropy(logits, labels)
        loss = torch.mean(losses)
        batch_loss = loss.item()
        avg_loss.update(batch_loss)

        _, preds = logits.max(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    pgd_accuracy = correct / (total + 1e-13)
    pgd_aua = pgd_accuracy
    logger.info(f'PGD Aua: {pgd_accuracy}')
    logger.info(f'PGD Loss: {avg_loss.get_metric()}')

    model.train()
    model.zero_grad()
    return pgd_accuracy,avg_loss


def evaluate(dev_loader, device, model):
    logger.info('Evaluating...')
    model.eval()
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()

    with torch.no_grad():
        for model_inputs, labels in dev_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits = model(**model_inputs, return_dict=False)[0]

            losses = F.cross_entropy(logits, labels)
            loss = torch.mean(losses)
            batch_loss = loss.item()
            avg_loss.update(batch_loss)

            _, preds = logits.max(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
    logger.info(f'Clean Aua: {accuracy}')
    logger.info(f'Clean Loss: {avg_loss.get_metric()}')
    return accuracy,avg_loss
    # logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')


def select_data(select_metric,select_ratio,num_labels,do_balance_labels, df, train_dataset):
    if "non_robust_" in select_metric:
        select_ratio = 1 - select_ratio

    df = sort_df_by_metric(df, select_metric)

    selected_label_nums = {}
    if do_balance_labels:
        df = df["id"]
        total_used_num_examples = int(df.shape[0] * select_ratio)
        every_label_num_examples = total_used_num_examples // num_labels
        sorted_data_indices = df.to_list()

        result_data_indices = []
        for i in range(num_labels):
            selected_label_nums[i] = 0
        for data_idx in sorted_data_indices:
            end_select = True
            cur_data_label = train_dataset.dataset[data_idx]['label']
            if selected_label_nums[cur_data_label] > every_label_num_examples:
                continue
            else:
                selected_label_nums[cur_data_label] += 1
                result_data_indices.append(data_idx)
            for i in range(num_labels):
                if not selected_label_nums[i] > every_label_num_examples:
                    end_select = False
            if end_select:
                break
        print("selected label indices:{}".format(selected_label_nums))
    # train_dataset.dataset = train_dataset.dataset.select(result_data_indices)

    else:
        df = df[:int(df.shape[0] * select_ratio)]
        df = df["id"]
        result_data_indices = df.to_list()

        for i in range(num_labels):
            selected_label_nums[i] = 0
        for data_idx in result_data_indices:
            selected_label_nums[train_dataset.dataset[data_idx]['label']] += 1
        print("selected label indices:{}".format(selected_label_nums))

    if "non_robust_" in select_metric:
        all_indices_set = set(range(len(sorted_data_indices)))
        non_robsut_set = set(result_data_indices)
        result_data_indices_set = all_indices_set-non_robsut_set
        result_data_indices = list(result_data_indices_set)

    return selected_label_nums,result_data_indices


def sort_df_by_metric(df, select_metric):
    """
    eg. loss_diff_mean_r+original_loss_mean
    :param df:
    :param select_metric:
    :return:
    """
    if "non_robust_" in select_metric:
        select_metric = select_metric[11:]
    if "+" in select_metric:
        select_metrics = select_metric.split("+")
        select_metric_x = select_metrics[0]
        select_metric_y = select_metrics[1]
        x_r = False
        y_r = False
        if select_metric_x[-2:]=="_r":
            x_r = True
            select_metric_x = select_metric_x[:-2]
        if select_metric_y[-2:]=="_r":
            y_r = True
            select_metric_y = select_metric_y[:-2]
        if x_r:
            x_destination = 0 # 取离0最近的
        else:
            x_destination = 1 # 取离1最近的

        if y_r:
            y_destination = 0 # 取离0最近的
        else:
            y_destination = 1

        df["sort_condition"] = (df[select_metric_x] - x_destination) * (df[select_metric_x] - x_destination) \
                               +\
                               (df[select_metric_y] - y_destination) * (df[select_metric_y] - y_destination)
        df = df.sort_values(by="sort_condition")

    else:
        if select_metric[-2:] == "_r":  # 从小到大
            df = df.sort_values(by=select_metric[:-2])
        elif select_metric == "top_left":  # (0,1)
            df["sort_condition"] = (df["loss_diff_mean"] - 0) * (df["loss_diff_mean"] - 0) + (df["loss_diff_std"] - 1) * (
                        df["loss_diff_std"] - 1)
            df = df.sort_values(by="sort_condition")

        elif select_metric == "top_right":  # 1,1)
            df["sort_condition"] = (df["loss_diff_mean"] - 1) * (df["loss_diff_mean"] - 1) + (df["loss_diff_std"] - 1) * (
                        df["loss_diff_std"] - 1)
            df = df.sort_values(by="sort_condition")

        elif select_metric == "bottom_left":  # (0,0)
            df["sort_condition"] = (df["loss_diff_mean"] - 0) * (df["loss_diff_mean"] - 0) + (df["loss_diff_std"] - 0) * (
                        df["loss_diff_std"] - 0)
            df = df.sort_values(by="sort_condition")

        elif select_metric == "bottom_right":  # (1,0)
            df["sort_condition"] = (df["loss_diff_mean"] - 1) * (df["loss_diff_mean"] - 1) + (df["loss_diff_std"] - 0) * (
                        df["loss_diff_std"] - 0)
            df = df.sort_values(by="sort_condition")

        else:  # 从大到小
            df = df.sort_values(by=select_metric, ascending=False)
    return df


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    finetune(args)