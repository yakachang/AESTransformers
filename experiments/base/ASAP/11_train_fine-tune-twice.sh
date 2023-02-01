#!/bin/bash

fold_name="fold_0"

pretrained="bert-base-cased"
max_len=512
lr=2e-5
batch_size=8
grad_acc=1

path_to_model="models/Original/base/sets/${fold_name}/lr${lr}-b${batch_size}a${grad_acc}"
fine_tuned_model="../ASAP++/models/Original/base/sets/set2/lr2e-5-b8a1-ada-fp16/bert-base-cased-512-mix-mod"
model_dir="${path_to_model}/${pretrained}-${max_len}-mod"
data_dir="../../../data/ASAP/folds/Original/${fold_name}"

if [[ -d "${model_dir}" ]]; then
    echo "${model_dir} exists! Skip training."
    exit
fi

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
    --model_name_or_path "${fine_tuned_model}" \
    --ignore_mismatched_sizes \
    --train_file "${data_dir}/train.json" \
    --validation_file "${data_dir}/dev.json" \
    --overwrite_output_dir \
    --output_dir "${model_dir}"
