# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : statistic_utils.py
@Project  : Robust_Data_new
@Time     : 2023/2/21 19:05
@Author   : Zhiheng Xi
"""


import numpy as np
from numpy import  *
import pandas as pd
import sys
import matplotlib.pyplot as plt


sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

def process_npy(statistics_path,len_dataset,dataset_name="glue",task_name="sst2",only_original_pred=True,
                use_normed_loss = False, use_delta_grad=False):
    """
    :param statistics_path:
    :param len_dataset:
    :param dataset_name:
    :param task_name:
    :param only_original_pred:
    :return:
    """
    test_npy = np.load(statistics_path, allow_pickle=True).item()
    intervals = list(test_npy)
    len_interval = len(test_npy)
    # len_dataset,_ = dataset_to_length_and_batch_size(dataset_name,task_name)
    new_data_loss_diff = [[] for i in range(len_dataset)]
    new_data_original_correctness = [0 for i in range(len_dataset)]
    new_data_flip_times = [0 for i in range(len_dataset)]
    new_data_original_loss = [[] for i in range(len_dataset)]
    new_data_perturbed_loss = [[] for i in range(len_dataset)]

    new_data_original_logit = [[] for i in range(len_dataset)]
    new_data_perturbed_logit = [[] for i in range(len_dataset)]
    new_data_logit_diff = [[] for i in range(len_dataset)]

    new_data_original_probability = [[] for i in range(len_dataset)]
    new_data_perturbed_probability = [[] for i in range(len_dataset)]
    new_data_probability_diff = [[] for i in range(len_dataset)]

    new_data_golden_label = [0 for i in range(len_dataset)]

    new_data_delta_grad = None
    if use_delta_grad:
        new_data_delta_grad = [[] for i in range(len_dataset)]
    tmp_idx = 0
    for interval in intervals:
        for data_idx in range(len_dataset):
            cur_data = test_npy[interval][data_idx]
            if only_original_pred:
                if cur_data["original_pred"]:
                    new_data_original_correctness[data_idx]+=1
                    new_data_original_loss[data_idx].append(cur_data["original_loss"])
                    new_data_perturbed_loss[data_idx].append(cur_data["after_perturb_loss"])
                    if cur_data.__contains__("original_logit"):
                        new_data_original_logit[data_idx].append(cur_data["original_logit"])
                        new_data_perturbed_logit[data_idx].append(cur_data["after_perturb_logit"])
                        new_data_logit_diff[data_idx].append(cur_data["logit_diff"])

                        new_data_original_probability[data_idx].append(cur_data["original_probability"])
                        new_data_perturbed_probability[data_idx].append(cur_data["after_perturb_probability"])
                        new_data_probability_diff[data_idx].append(cur_data["probability_diff"])

                        new_data_golden_label[data_idx] = cur_data["golden_label"]

                    if not use_normed_loss:
                        new_data_loss_diff[data_idx].append(cur_data["loss_diff"])
                    else:
                        new_data_loss_diff[data_idx].append(cur_data["normed_loss_diff"])
                    if use_delta_grad:
                        new_data_delta_grad[data_idx].append(cur_data["delta_grad"])

                    if cur_data["original_pred"] != cur_data["after_perturb_pred"]:
                        new_data_flip_times[data_idx]+=1


            else:

                if cur_data["original_pred"]:
                    new_data_original_correctness[data_idx] += 1
                new_data_original_loss[data_idx].append(cur_data["original_loss"])
                new_data_perturbed_loss[data_idx].append(cur_data["after_perturb_loss"])

                if cur_data.__contains__("original_logit"):
                    new_data_original_logit[data_idx].append(cur_data["original_logit"])
                    new_data_perturbed_logit[data_idx].append(cur_data["after_perturb_logit"])
                    new_data_logit_diff[data_idx].append(cur_data["logit_diff"])

                    new_data_original_probability[data_idx].append(cur_data["original_probability"])
                    new_data_perturbed_probability[data_idx].append(cur_data["after_perturb_probability"])
                    new_data_probability_diff[data_idx].append(cur_data["probability_diff"])

                    new_data_golden_label[data_idx] = cur_data["golden_label"]

                if not use_normed_loss:
                    new_data_loss_diff[data_idx].append(cur_data["loss_diff"])
                else:
                    new_data_loss_diff[data_idx].append(cur_data["normed_loss_diff"])
                if use_delta_grad:
                    new_data_delta_grad[data_idx].append(cur_data["delta_grad"])
                if cur_data["original_pred"] != cur_data["after_perturb_pred"]:
                    new_data_flip_times[data_idx] += 1
            # print("tmp_idx:{}".format(tmp_idx))
            tmp_idx+=1


    new_data_original_correctness = [new_data_original_correctness[i]/len_interval for i in range(len_dataset)]
    return new_data_loss_diff,new_data_original_correctness,new_data_flip_times,\
           new_data_delta_grad,new_data_original_loss,new_data_perturbed_loss,new_data_original_logit,\
           new_data_perturbed_logit,new_data_logit_diff ,new_data_original_probability,\
           new_data_perturbed_probability,new_data_probability_diff , new_data_golden_label


def data_with_metrics(data_loss_diff, data_original_correctness,
                      data_flip_times,data_delta_grad=None,new_data_original_loss=None,
                      new_data_perturbed_loss=None,
                      new_data_original_logit=None,
                      new_data_perturbed_logit=None,
                      new_data_logit_diff=None,
                      new_data_original_probability=None,
                      new_data_perturbed_probability=None,
                      new_data_probability_diff=None,new_data_golden_label=None,
                      require_original_data = False,
                      do_norm=False,do_norm_for_prob=False):
    """
    transform to dataframe
    """
    result = []
    for i in range(len(data_loss_diff)):
        cur_res = {}
        if data_loss_diff[i] == []:
            continue
        else:
            cur_res["id"] = i
            cur_res["golden_label"] = new_data_golden_label[i]
            # correctness & flip times
            cur_res["original_correctness"] = data_original_correctness[i]
            cur_res["flip_times"] = data_flip_times[i]
            # loss
            cur_res["loss_diff_mean"] = mean(data_loss_diff[i])
            cur_res["loss_diff_std"] = std(data_loss_diff[i])
            cur_res["original_loss_mean"] = mean(new_data_original_loss[i])
            cur_res["original_loss_std"] = std(new_data_original_loss[i])
            cur_res["perturbed_loss_mean"] = mean(new_data_perturbed_loss[i])
            cur_res["perturbed_loss_std"] = std(new_data_perturbed_loss[i])
            # logit
            cur_res["original_logit_mean"] = mean(new_data_original_logit[i])
            cur_res["original_logit_std"] = std(new_data_original_logit[i])
            cur_res["perturbed_logit_mean"] = mean(new_data_perturbed_logit[i])
            cur_res["perturbed_logit_std"] = std(new_data_perturbed_logit[i])
            cur_res["logit_diff_mean"] = mean(new_data_logit_diff[i])
            cur_res["logit_diff_std"] = std(new_data_logit_diff[i])
            # prob
            cur_res["original_probability_mean"] = mean(new_data_original_probability[i])
            cur_res["original_probability_std"] = std(new_data_original_probability[i])
            cur_res["perturbed_probability_mean"] = mean(new_data_perturbed_probability[i])
            cur_res["perturbed_probability_std"] = std(new_data_perturbed_probability[i])
            cur_res["probability_diff_mean"] = mean(new_data_probability_diff[i])
            cur_res["probability_diff_std"] = std(new_data_probability_diff[i])

            if data_delta_grad!=None:
                cur_res["delta_grad_mean"] = mean(data_delta_grad[i])
                cur_res["delta_grad_std"] = std(data_delta_grad[i])
            result.append(cur_res)
    dataframe = pd.DataFrame(result)
    if do_norm:
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        dataframe["loss_diff_mean"] = dataframe[['loss_diff_mean']].apply(max_min_scaler)
        dataframe["loss_diff_std"] = dataframe[['loss_diff_std']].apply(max_min_scaler)
        dataframe["original_loss_mean"] = dataframe[["original_loss_mean"]].apply(max_min_scaler)
        dataframe["original_loss_std"] = dataframe[["original_loss_std"]].apply(max_min_scaler)
        dataframe["perturbed_loss_mean"] = dataframe[["perturbed_loss_mean"]].apply(max_min_scaler)
        dataframe["perturbed_loss_std"] = dataframe[["perturbed_loss_std"]].apply(max_min_scaler)

        dataframe["original_logit_mean"] = dataframe[["original_logit_mean"]].apply(max_min_scaler)
        dataframe["original_logit_std"] = dataframe[["original_logit_std"]].apply(max_min_scaler)
        dataframe["perturbed_logit_mean"] = dataframe[["perturbed_logit_mean"]].apply(max_min_scaler)
        dataframe["perturbed_logit_std"] = dataframe[["perturbed_logit_std"]].apply(max_min_scaler)
        dataframe["logit_diff_mean"] = dataframe[["logit_diff_mean"]].apply(max_min_scaler)
        dataframe["logit_diff_std"] = dataframe[["logit_diff_std"]].apply(max_min_scaler)

    if do_norm_for_prob:
        dataframe["original_probability_mean"] = dataframe[["original_probability_mean"]].apply(max_min_scaler)
        dataframe["original_probability_std"] = dataframe[["original_probability_std"]].apply(max_min_scaler)
        dataframe["perturbed_probability_mean"] = dataframe[["perturbed_probability_mean"]].apply(max_min_scaler)
        dataframe["perturbed_probability_std"] = dataframe[["perturbed_probability_std"]].apply(max_min_scaler)
        dataframe["probability_diff_mean"] = dataframe[["probability_diff_mean"]].apply(max_min_scaler)
        dataframe["probability_diff_std"] = dataframe[["probability_diff_std"]].apply(max_min_scaler)



    if require_original_data:
        return dataframe,data_loss_diff,data_original_correctness,data_flip_times,data_delta_grad,new_data_original_loss,new_data_perturbed_loss
    return dataframe

