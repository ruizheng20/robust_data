# coding=utf-8
"""
fuckasd asda osid oasd oa sdoiha dsoih asdh ashd aihs doih
"""
import os
import csv
os.environ['TFHUB_CACHE_DIR'] = '/root/tfhub_modules'
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import argparse
import logging
from transformers import AutoConfig, AutoTokenizer
from models.modeliing_bert import BertForSequenceClassification
# from modeling_roberta_ER import RobertaForSequenceClassification

from textattack import Attacker
from textattack import AttackArgs

from textattack.models.wrappers import HuggingFaceModelWrapper
from attack_utils import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult

logger = logging.getLogger(__name__)


from attack.build_attacker_utils import build_weak_attacker, build_english_attacker

def attack_parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--attack_method', type=str, default='textfooler')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default="sst2", type=str)
    parser.add_argument("--model_name", default='/root/Early_Robust/baselines/fine_tune/saved_models/epoch0', type=str)
    parser.add_argument("--model_type", default='bert-base-uncased', type=str)
    parser.add_argument('--training_type', type=str, default='fine_tune')
    parser.add_argument('--valid', type=str, default='validation')
    parser.add_argument("--num_examples", default=1000, type=int)
    parser.add_argument("--results_file", default='attack_log.csv', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--attack_seed", default=42, type=int)
    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.15, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)

    # additional
    parser.add_argument('--max_seq_length', default=128, type=int, help='')
    parser.add_argument('--random_select', default=False, type=bool, help='')
    # parser.add_argument('--num_train_examples_per_class', default=-1, type=int, help='')
    parser.add_argument('--num_train_examples_ratio', default=1.0, type=float, help='')
    parser.add_argument('--data_indices_file', default=None, type=str, help='')
    parser.add_argument('--num_labels', type=int, default=2)
    # parser.add_argument('--attack_method', type=str, default="textfooler")

    args = parser.parse_args()
    return args


def do_attack_with_model(model,tokenizer,
                         dataset_name="glue",task_name="sst2",
                         valid="validation",attack_method="textfooer",
                         num_examples=1000,attack_seed=42,results_file="attack.csv",seed=42,model_name="bert-base-uncased",
                         log_format=None,data_row=None):


    if dataset_name == 'imdb' or dataset_name == 'ag_news' or dataset_name=="SetFit/20_newsgroups":
        task_name=None
        valid = "test"
    elif task_name=="mnli":
        valid = "validation_matched"
        output_mode = "classification"
    else:
        valid = "validation"
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    if attack_method == "bertattack":
        neighbour_vocab_size = 50
        modeify_ratio = 0.9
        sentence_similarity = 0.2
        attack = build_weak_attacker(neighbour_vocab_size,sentence_similarity, model_wrapper,attack_method)
    elif attack_method == "textfooler" and dataset_name == "SetFit/20_newsgroups":
        neighbour_vocab_size = 10
        modeify_ratio = 0.9
        sentence_similarity = 0.85
        attack = build_weak_attacker(neighbour_vocab_size,sentence_similarity, model_wrapper, attack_method)
    else:
        attack = build_english_attacker( model_wrapper, attack_method)
    # attack = build_english_attacker(args, model_wrapper)
    # dataset = utils.Huggingface_dataset(args,tokenizer,dataset_name,
    #                              subset="sst2" if task_name=="sst-2" else task_name
    #                              , split=valid)
    dataset = HuggingFaceDataset(dataset_name,
                                 subset="sst2" if task_name == "sst-2" else task_name
                                 , split="test" if dataset_name == "SetFit/20_newsgroups" else valid)

    logger.info("shuffled attack set!")
    # dataset.shuffle()
    # dataset._dataset =
    if dataset_name=="glue" and task_name=="sst2":
        attack_args = AttackArgs(num_examples=num_examples, disable_stdout=True, random_seed=attack_seed,shuffle=False)
    else:
        attack_args = AttackArgs(num_examples=num_examples, disable_stdout=True, random_seed=attack_seed,shuffle=True)
    attacker = Attacker(attack, dataset, attack_args)

    num_results = 0
    num_successes = 0
    num_failures = 0

    printed = 0
    for result in attacker.attack_dataset():
        if printed==0:
            logger.info(result)
            printed+=1
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

    logger.info("[Succeeded/Failed/Total] {} / {} / {}".format(num_successes, num_failures, num_results))

    # compute metric
    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results
    attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy

    if not os.path.exists(results_file):
        out_csv = open(results_file, 'a', encoding='utf-8', newline="")
        csv_writer = csv.writer(out_csv)
        cur_row = [i for i in log_format]
        cur_row.append("original_accuracy")
        cur_row.append("accuracy_under_attack")
        cur_row.append("attack_succ")
        csv_writer.writerow(cur_row)
        out_csv.close()
    out_csv = open(results_file, 'a', encoding='utf-8', newline="")
    csv_writer = csv.writer(out_csv)
    if log_format==None:
        csv_writer.writerow([model_name,seed, original_accuracy, accuracy_under_attack, attack_succ])
    else:

        cur_row = data_row
        cur_row.append(original_accuracy)
        cur_row.append(accuracy_under_attack)
        cur_row.append(attack_succ)
        csv_writer.writerow(cur_row)
    out_csv.close()
    logger.info(
        "[Accuracy/Aua/Attack_success] {} / {} / {}".format(original_accuracy, accuracy_under_attack, attack_succ))


def do_attack(args):
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news' or args.dataset_name=="SetFit/20_newsgroups":
        args.task_name=None
        args.valid = "test"
    else:
        args.valid = "validation"
    config = AutoConfig.from_pretrained(args.model_name)
    if args.model_type == "bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)

    elif args.model_type == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    log_format = []
    data_row=[]

    do_attack_with_model( model, tokenizer,
                             dataset_name=args.dataset_name,
                             task_name=args.task_name, valid=args.valid,
                             attack_method=args.attack_method,
                             num_examples=args.num_examples, attack_seed=args.attack_seed,
                             results_file=args.results_file,
                             seed=args.seed,
                             model_name=args.model_name,log_format=log_format,data_row=data_row
                             )



def main():
    args = attack_parse_args()
    print(args)
    do_attack(args)


if __name__ == "__main__":
    main()
