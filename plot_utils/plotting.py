# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : plotting.py.py
@Project  : Robust_Data_new
@Time     : 2023/2/24 14:40
@Author   : Zhiheng Xi
"""
import seaborn as sns
import numpy as np
from numpy import  *
import pandas as pd
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from finetune_with_select_data import select_data,sort_df_by_metric



from statistic_utils import *

def dataset_to_length_and_batch_size(dataset_name="glue",task_name="sst2"):
    if dataset_name == "imdb":
        len_dataset = 25000
        per_device_batch_size = 8
        return len_dataset,per_device_batch_size
    elif dataset_name == "ag_news":
        len_dataset = 120000
        per_device_batch_size = 8
        return len_dataset,per_device_batch_size
    elif dataset_name == "SetFit/20_newsgroups":
        len_dataset = 16019
        per_device_batch_size = 8
        return len_dataset, per_device_batch_size

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




# Based on https://github.com/allenai/cartography
def plot_map(dataframe,max_instances_to_plot,hue_metric="original_correctness",main_metric="loss_diff_mean",other_metric="loss_diff_std",do_round=True,show_hist=True,model="BERT",dataset="SST2"):


    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')
    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # normalize
    dataframe = dataframe.assign(original_corr_frac = lambda d: d.original_correctness / d.original_correctness.max())
    dataframe['original_correctness'] = [f"{x:.1f}" for x in dataframe['original_corr_frac']]

    # main_metric = 'loss_diff_mean'
    # other_metric = 'loss_diff_std'
    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues <= 11 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])
    #
    # colors = [
    #     "#ea7a66",
    #     "#ed5f4b",
    #     "#f14924",
    #     "#d13123",
    #     "#ea514c",
    #     "#983025",
    #     "#2b66c0",
    #     "#4855b5",
    #     "#2e3178",
    #     "#92b36f",
    #     "#59b351"
    # ]
    # colors.reverse()
    # pal = sns.color_palette(
    #     colors
    # )

    pal = sns.diverging_palette(260, 20, n=num_hues, sep=20, center="dark")
    # pal = sns.diverging_palette(15,320, n=num_hues, sep=20, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    # else:
    #     plot.legend(fancybox=True, shadow=True, ncol=1)
    if show_hist:
        # plot.set_title("{}-{} Robust Data Map".format(dataset,model))

        # histograms
        # ax1 =
        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        # plott0 = dataframe.hist(column=['Perturbed Loss Mean'], ax=ax1, color='#622a87')
        plott0 = dataframe.hist(column=['Sensitivity'], ax=ax1, color='#816e9c')
        plott0[0].set_title('')
        plott0[0].set_xlabel('Sensitivity')
        plott0[0].set_ylabel('Density')

        # plott1 = dataframe.hist(column=['Perturbed Loss Std'], ax=ax2, color='teal')
        plott1 = dataframe.hist(column=['Variability'], ax=ax2, color='#6989b9')
        plott1[0].set_title('')
        plott1[0].set_xlabel('Variability')
        plott1[0].set_ylabel('Density')

        # plott2 = dataframe.hist(column=['Flip Rate'], ax=ax3, color='#86bf91')
        plott2 = dataframe.hist(column=['Flip Rate'], ax=ax3, color='#a9c1a3')
        plott2[0].set_title('')
        plott2[0].set_xlabel('Flip Rate')
        plott2[0].set_ylabel('Density')


    plt.savefig("{}.pdf".format(dataset),pad_inches=0)
    plt.show()

    print()

if __name__ == '__main__':

    """
    Choose your dataset and specify your statistic path, and plot
    """

    len_dataset,_ = dataset_to_length_and_batch_size("glue","sst2")
    your_path = "your_path.npy"
    new_data_loss_diff, new_data_original_correctness, new_data_flip_times, \
    new_data_delta_grad, new_data_original_loss, new_data_perturbed_loss, new_data_original_logit, \
    new_data_perturbed_logit, new_data_logit_diff, new_data_original_probability, \
    new_data_perturbed_probability, new_data_probability_diff,new_data_golden_label= process_npy(
        your_path, # todo set your statistic path here
        len_dataset=len_dataset,use_normed_loss=False,use_delta_grad=False,
        only_original_pred=False
                               )
    df = data_with_metrics( new_data_loss_diff,
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
                            do_norm=True)
    df["Flip Rate"] = df["flip_times"] / 10
    df["Perturbed Loss Mean"] = df["perturbed_loss_mean"]
    df["Perturbed Loss Std"] = df["perturbed_loss_std"]
    df["Sensitivity"] = df["perturbed_loss_mean"]
    df["Variability"] = df["perturbed_loss_std"]

    plot_map(df,
             # int(df.shape[0]),
             120000,
             hue_metric="Flip Rate",
             main_metric="Sensitivity",
             other_metric="Variability",
             model="BERT",
             dataset="AG_NEWS"
             )

    pass

