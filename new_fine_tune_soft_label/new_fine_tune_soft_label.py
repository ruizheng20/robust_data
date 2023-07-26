# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : new_fine_tune_soft_label.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 19:20
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
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)
from models.modeliing_bert import BertForSequenceClassification
from models.modeling_roberta import RobertaForSequenceClassification
from torch.utils.tensorboard import SummaryWriter

import utils as utils
# from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer # modified
# from modeling_utils import PreTrainedModel
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import statistic_utils as statistic_utils
from finetune_with_select_data import save_model_one_step,save_model_one_epoch,generate_data_indices,do_textattack_attack,do_pgd_attack,evaluate,select_data,sort_df_by_metric
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
    parser.add_argument('--output_dir',type=str,default='/root/Robust_Data_Outputs/baselines/fine_tune/saved_models')

    parser.add_argument('--reinit_classifier', default=0, type=int, help="")
    parser.add_argument('--freeze_bert', default=0, type=int, help="")

    # others
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--attack_every_epoch', default=0,type=int,help="10:攻击train dataset1；20：攻击 train dataset2；否则攻击dev dataset")
    parser.add_argument('--attack_every_step', default=0,type=int)
    parser.add_argument('-f', '--not_force_overwrite',action="store_true") # 只有传入了这个参数才会是true
    parser.add_argument('--cal_time',action="store_true")

    # few-shot setting
    parser.add_argument('--random_select', default=0, type=int, help='')
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
    parser.add_argument('--pgd_step', type=int, default=5)
    parser.add_argument('--pgd_lr', type=float, default=0.05)
    # freelb
    parser.add_argument('--pgd_adv_steps', type=int, default=5)
    parser.add_argument('--pgd_adv_steps2', type=int, default=10)
    parser.add_argument('--pgd_adv_lr', type=float, default=0.03)
    parser.add_argument('--pgd_adv_lr2', type=float, default=0.03)
    parser.add_argument('--do_pgd_training', type=int, default=0)

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

    # new finetune
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--soft_label', type=float, default=1)
    parser.add_argument('--log_file', type=str, default="")
    parser.add_argument('--non_robust_type', type=str, default="soft")



    args = parser.parse_args()
    # if args.balance_labels:
    #     print("s")
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '..'
    print(args.__dict__)

    return args

def set_seed(seed:int):
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

data_indices_to_new_id={}
new_id_to_data_indices={}

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

    tensorboard_path = "/root/Robust_Data/runs/new_finetune_soft_label_from_init_not_symmetry"+args.output_dir[args.output_dir.rfind("/"):]
    writer = SummaryWriter(tensorboard_path)

    # args.writer = writer

    # prepare datasets
    # logger.info(utils.say())
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news' or args.dataset_name=="SetFit/20_newsgroups":
        args.task_name = None
        args.valid = "test"


    train_dataset = utils.Huggingface_dataset_with_data_id(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)

    assert args.statistics_source != None
    # 现在是两类都选择同样的ratio，一半一半，即30%+30%
    # todo  我想的是整个数据集，然后比如前30%是鲁棒的，其他是不鲁棒的

    result_data_indices, selected_label_nums = generate_data_indices(args, train_dataset,args.select_metric,args.select_ratio)
    # if args.select_metric[-2:] == "_r":
    #     select_metric2 = args.select_metric[:-2]
    # else:
    #     select_metric2 = args.select_metric + "_r"
    # result_data_indices2, selected_label_nums2 = generate_data_indices(args, train_dataset, select_metric2,
    #                                                                          1.0-args.select_ratio)
    # train_dataset.dataset = train_dataset.dataset.select(result_data_indices+result_data_indices2)


    # result_data_indices_new = result_data_indices+result_data_indices2
    # result_data_indices_new = [i for i in range(train_dataset.__len__())]
    # for i in range(len(result_data_indices_new)):
    #     data_indices_to_new_id[result_data_indices_new[i]] = i
    #     new_id_to_data_indices[i] = result_data_indices_new[i]
        # pass
    # if args.dataset_name=="imdb" or args.dataset_name=="ag_news":
    #     for i in range(len(result_data_indices)):
    #         data_indices_to_new_id[result_data_indices[i]] = i
    #         new_id_to_data_indices[i] = result_data_indices[i]
    #     train_dataset.updata_idx(data_indices_to_new_id,new_id_to_data_indices)
    # result_data_indices = [i for i in range(len(result_data_indices))]
    # old_to_new_dict = {}


    if args.show_data > 0:
        import csv
        print("Metric:{}".format(args.select_metric))
        for i in range(args.show_data):
            print("sentence: {}, label: {}".format(train_dataset.dataset["sentence"][i],train_dataset.dataset["label"][i]))
            print()
            print()
            show_data_dir = "/root/Robust_Data/analysis_experiments/show_data/"
            show_data_file = "/root/Robust_Data/analysis_experiments/show_data/show_data.csv"
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
            cur_row.append(train_dataset.dataset["sentence"][i])
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

    log_format = ["statistics_source",
                  "select_metric",
                  "select_ratio", "ratio_proportion",
                  "selected_label_nums",
                  "lr",
                  "seed", "epochs",
                  "pgd_step", "pgd_lr",
                  "clean", "pgd_aua", "attack_method","beta","soft_label"
                  ]

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
            global_step = train_one_epoch_new(
                args,
                avg_loss,
                device,
                epoch,
                model,
                optimizer,
                scheduler,
                train_loader,
                global_step,
                save_steps,
                tokenizer,
                result_data_indices,
                beta=args.beta,
                writer=writer,
                dev_loader=dev_loader)
            # save model
            # save_model_one_epoch(args, epoch, model, output_dir, tokenizer)
            # eval model
            if args.do_eval and not args.cal_time:
                accuracy,clean_loss = evaluate(dev_loader, device, model)
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

                        args.clean_loss =clean_loss.get_metric()
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
                                         attack_every_step=True,
                                         log_format=[i for i in log_format]

                                         )

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
                                     attack_every_step=False,
                                     log_format=[i for i in log_format]
                                     )

        if (not args.attack_every_epoch and not args.attack_every_step and args.do_attack):
            if args.do_eval and not args.cal_time:
                args.attack_dataset_metric = "test_set"
                loader = dev_loader
                clean, clean_loss = evaluate(loader, device, model)
                args.clean = clean
                args.clean_loss = clean_loss
            else:
                args.clean = 0
                args.clean_loss = 0
            if args.do_pgd_attack:
                pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                                 adv_steps=args.pgd_step,
                                                 adv_lr=args.pgd_lr,
                                                 adv_init_mag=args.adv_init_mag,
                                                 adv_max_norm=args.adv_max_norm,
                                                 adv_norm_type=args.adv_norm_type
                                                 )
                args.pgd_aua = pgd_aua
                args.pgd_loss = pgd_loss
            else:
                args.pgd_aua = 0
                args.pgd_loss = 0
            optimizer.zero_grad()
            do_textattack_attack(args, model, tokenizer,
                                 do_attack=args.do_attack,
                                 attack_seed=42,
                                 attack_all=args.attack_all,
                                 attack_method="textfooler",
                                 attack_every_epoch=args.attack_every_epoch,
                                 log_format=[i for i in log_format]
                                 )
        elif (not args.attack_every_epoch and not args.attack_every_step and not args.do_attack and args.do_pgd_attack):
            pgd_aua,pgd_loss = do_pgd_attack( dev_loader, device, model,
                                             adv_steps=args.pgd_step,
                                             adv_lr=args.pgd_lr,
                                             adv_init_mag=args.adv_init_mag,
                                             adv_max_norm=args.adv_max_norm,
                                             adv_norm_type=args.adv_norm_type,
                                             )
            optimizer.zero_grad()
            args.pgd_aua = pgd_aua
            args.pgd_loss = pgd_loss

    except KeyboardInterrupt:
        logger.info('Interrupted...')



def calculate_perturbed_loss_one_batch(args, model, model_inputs, labels,adv_steps,adv_lr):
    model.eval()
    # perturbed loss计算
    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)
    correct = 0
    total = 0

    delta = torch.zeros_like(embedding_init)
    for astep in range(adv_steps):
        # 0. forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch, return_dict=False)[0]
        _, preds = logits.max(dim=-1)
        # 1.
        losses = F.cross_entropy(logits, labels.squeeze(-1))
        # losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / adv_steps
        loss.backward()

        if astep == adv_steps - 1:
            losses = F.cross_entropy(logits, labels.squeeze(-1),reduction="none")
            _, preds = logits.max(dim=-1)
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
            model.train()
            model.zero_grad()
            embedding_init = word_embedding_layer(input_ids)
            delta.requires_grad = False
            return losses,correct,total
            # pass

        # 2. get gradient on delta
        delta_grad = delta.grad.clone().detach()
        # 3. update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + adv_lr * delta_grad / denorm).detach()  # detach 截断反向传播?
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
            delta = (delta + adv_lr * delta_grad / denorm).detach()
        model.zero_grad()
        embedding_init = word_embedding_layer(input_ids)  # 重新初始化embedding



def train_one_epoch_new(args, avg_loss, device, epoch, model, optimizer, scheduler, train_loader, global_step, save_steps, tokenizer,result_data_indices,beta=1,writer=None,dev_loader=None):



    pbar = tqdm(train_loader,ncols=100)
    for model_inputs, labels_idx in pbar:
        labels = torch.tensor([i[0] for i in labels_idx])
        indices = torch.tensor([i[1] for i in labels_idx])
        selected_set = set(result_data_indices)
        data_selected  = torch.tensor([1 if int(i) in selected_set  else 0 for i in indices])
        data_not_selected = torch.tensor([0 if int(i) in selected_set  else 1 for i in indices])
        if save_steps > 0 and global_step % save_steps ==0:
            save_model_one_step(args,global_step,model,output_dir=args.output_dir,tokenizer=tokenizer)
        batch_loss = 0
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        logits = model(**model_inputs, return_dict=False)[0]
        _, preds = logits.max(dim=-1)

        losses_hard = F.cross_entropy(logits, labels.squeeze(-1),reduction="none")
        losses_soft = SoftCrossEntropy(logits,labels.squeeze(-1),reduction="none",soft_label=args.soft_label,device=device)
        probs = F.softmax(logits,dim=1)

        soft_coef = beta * (torch.tensor([1 for i in range(len(labels))]))
        hard_loss = data_not_selected.to(device).mul(losses_hard)
        soft_loss = data_selected.to(device).mul(soft_coef.to(device)).mul(losses_soft)
        losses = soft_loss+hard_loss

        loss = torch.mean(losses)
        # loss2  = model(**model_inputs,return_dict=False)
        batch_loss = loss.item()



        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        avg_loss.update(batch_loss)

        writer.add_scalars("train_loss", {"whole_loss":avg_loss.get_metric()},global_step)

        pbar.set_description(f'epoch: {epoch: d}, '
                             f'loss: {avg_loss.get_metric(): 0.4f}, '
                             f'lr: {optimizer.param_groups[0]["lr"]: .3e},'
                             )
        global_step+=1

    return global_step


def calculate_dev_set_perturbed_loss(args, dev_loader, device, model):
    model.eval()
    pbar = tqdm(dev_loader)
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        model.zero_grad()
        perturbed_losses,cur_correct,cur_total = calculate_perturbed_loss_one_batch(args=args, model=model, model_inputs=model_inputs,
                                                              labels=labels,adv_steps=args.pgd_step,adv_lr=args.pgd_lr)
        batch_loss = torch.mean(perturbed_losses).item()
        avg_loss.update(batch_loss)
        correct+=cur_correct
        total+=cur_total
    pgd_accuracy = correct / (total + 1e-13)
    pgd_aua = pgd_accuracy
    logger.info(f'test perturbed Aua: {pgd_accuracy}')
    logger.info(f'test perturbed Loss: {avg_loss.get_metric()}')
    model.train()
    model.zero_grad()
    return pgd_aua,avg_loss


def SoftCrossEntropy(inputs, target, reduction='none',soft_label=1,device="cuda"):
    """
    soft label的loss实现
    :param inputs:
    :param target:
    :param reduction:
    :param soft_label: golden label的值，剩下的值被其他标签平分
    :param device:
    :return:
    """
    log_likelihood = -F.log_softmax(inputs, dim=1)
    num_labels = inputs.shape[1]
    batch = inputs.shape[0]

    new_target = F.one_hot(target,num_labels).to(device)
    inverse_target = (torch.ones(inputs.shape).to(device) - new_target).to(device)

    new_target = new_target * soft_label + inverse_target * ((1-soft_label) / (num_labels-1))
    losses = torch.sum(torch.mul(log_likelihood, new_target),dim=1)
    if reduction == 'average':
        losses = torch.sum(losses) / batch
    elif reduction == "none":
        return losses
    elif reduction=="sum":
        losses = torch.sum(losses)

    return losses


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    finetune(args)
