CUDA_VISIBLE_DEVICES=0  nohup  python new_fine_tune_flooding.py      \
 --model_name bert-base-uncased      \
 --dataset_name glue      \
 --task_name sst2       \
 --max_seq_length 128    \
 --bsz 32       \
 --lr 2e-5       \
 --seed 42       \
 --epochs 3       \
 --num_labels 2     \
 --statistics_source your_path.npy      \
 --select_metric perturbed_loss_mean_r      \
 --select_ratio 0.3      \
 --output_dir /root/Robust_Data_new/new_fine_tune_flooding/sst2_outputs      \
 --do_attack --attack_all      \
 --results_file /root/Robust_Data_new/new_fine_tune_flooding/sst2_results.csv      \
 --do_balance_labels 1      \
 --num_examples 1000       \
 --pgd_step 8         \
 --pgd_lr 0.06        \
 --use_cur_preds 0      \
 --with_untrained_model 0        \
 --only_original_pred 0  \
 --attack_every_epoch 0 \
 --attack_every_step 0 \
 --save_steps -1 \
 --beta 1 --b 0.2        \
    > sst2.log 2>&1 &

