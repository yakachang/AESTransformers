#!/bin/bash

# "Original", "Max-Min-Scaling"
data_type="Balanced-Data-Max"

for fold_name in 'fold_0' 'fold_1' 'fold_2' 'fold_3' 'fold_4'; do

    # fold_name="fold_0"
    pretrained="bert-base-cased"
    max_len=512
    lr=2e-5
    batch_size=8
    grad_acc=1

    setting="lr${lr}-b${batch_size}a${grad_acc}-ada-fp16"
    path_to_model="models/${data_type}/base/${fold_name}/${setting}"
    model_dir="${path_to_model}/${pretrained}-${max_len}-mod"
    data_dir="../../../data/ASAP/folds/${data_type}/${fold_name}"

    python ../../../run_classifier.py \
        --per_device_train_batch_size "${batch_size}" \
        --per_device_eval_batch_size "${batch_size}" \
        --gradient_accumulation_steps "${grad_acc}" \
        --learning_rate "${lr}" \
        --num_train_epochs 20.0 \
        --logging_steps 100 \
        --evaluation_strategy="epoch" \
        --save_strategy="epoch" \
        --load_best_model_at_end \
        --overwrite_cache \
        --max_seq_length "${max_len}" \
        --do_train \
        --do_eval \
        --adafactor \
        --fp16 \
        --fp16_full_eval \
        --model_name_or_path "${pretrained}" \
        --train_file "${data_dir}/train.json" \
        --validation_file "${data_dir}/dev.json" \
        --overwrite_output_dir \
        --output_dir "${model_dir}"
done
