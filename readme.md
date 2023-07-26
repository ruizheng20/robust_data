## Codes for "Characterizing the Impacts of Instances on Robustness"
###Basic Infomation
- paper: https://aclanthology.org/2023.findings-acl.146.pdf
- Authors: Rui Zheng*, Zhiheng Xi*, Qin Liu, Wenbin Lai, Tao Gui, Qi Zhang, Xuanjing Huang, Jin Ma, Ying Shan, Weifeng Ge.
### Usage
- Collect robust statistics of training dataset
```shell script
python data_statistics.py
```

- The data is saved to the following path (see in `data_statistics.py`):
```python
np.save('robust_statistics_model{}_dataset{}_task{}_seed{}_shuffle{}_len{}_adv_steps{}_adv_lr{}_epoch{}_lr{}_interval{}_with_untrained_model{}_use_cur_preds{}.npy'
    .format(args.model_name,
            args.dataset_name,
            args.task_name,
            args.seed,
            args.do_train_shuffle,
            args.dataset_len,
            args.adv_steps,
            args.adv_lr,args.epochs,
            args.lr,
            args.statistic_interval,
            args.with_untrained_model,
            args.use_cur_preds
            ),
    robust_statistics_dict)
```

- Draw plots to show data robustness (remember to set your statistic path in the file)
```shell script
cd plot_utils
python plotting.py
```

- Run Flooding method with robust data
```shell script
cd new_fine_tune_flooding
sh run_sst2_new_finetune_flooding.sh
```

- Run Soft Label method with robust data
```shell script
cd new_fine_tune_flooding
sh run_sst2_new_finetune_soft_label.sh
```

## Plot & Performance

- Robust Data Map

<img src="https://spring-security.oss-cn-beijing.aliyuncs.com/img/image-20230726195217008.png" alt="image-20230726195217008" style="zoom:50%;" />

- Final Performance

<img src="https://spring-security.oss-cn-beijing.aliyuncs.com/img/image-20230726195246173.png" alt="image-20230726195246173" style="zoom:50%;" />

- See more analysis in our paper!

## Citation

```
@inproceedings{zheng2023characterizing,
  title={Characterizing the Impacts of Instances on Robustness},
  author={Zheng, Rui and Xi, Zhiheng and Liu, Qin and Lai, Wenbin and Gui, Tao and Zhang, Qi and Huang, Xuan-Jing and Ma, Jin and Shan, Ying and Ge, Weifeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={2314--2332},
  year={2023}
}
```