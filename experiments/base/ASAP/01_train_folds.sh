#!/bin/bash

batch_size=8
grad_acc=1
data_dir="../../../data/ASAP/folds/Min-Max-Scaling/fold_0"
model_dir="models/Min-Max-Scaling-fold_0"

python ../../../run_classifier.py \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size "${batch_size}" \
    --gradient_accumulation_steps "${grad_acc}" \
    --num_train_epochs 20.0 \
    --logging_steps 100 \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --load_best_model_at_end \
    --overwrite_cache \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --model_name_or_path bert-base-cased \
    --train_file "${data_dir}/train.json" \
    --validation_file "${data_dir}/dev.json" \
    --overwrite_output_dir \
    --output_dir "${model_dir}"
