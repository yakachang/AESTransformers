#!/bin/bash

set -ex

seed=3435
lr=2e-5
max_len=512
batch_size=4
grad_acc=1
max_epoch=20
# patience=5
pretrained="google/flan-t5-small"
source_key="text"
target_key="label"

for fold_id in "fold_0" "fold_1" "fold_2" "fold_3" "fold_4";
do
    data_dir="../../data/ASAP++/Multi-Task/folds_p1-2/${fold_id}"
    train_file="${data_dir}/train.json"
    dev_file="${data_dir}/dev.json"
    model_dir="test-mod"

    if [[ -d "${model_dir}" ]]; then
        echo "${model_dir} exists! Skip training."
        exit
    fi

    # --do_train \
    # --do_eval \
    # --overwrite_output_dir \
    python ../../run_seq2seq.py \
        --seed ${seed} \
        --learning_rate "${lr}" \
        --max_source_length=${max_len} \
        --per_device_train_batch_size=${batch_size} \
        --per_device_eval_batch_size=${batch_size} \
        --gradient_accumulation_steps=${grad_acc} \
        --num_train_epochs=${max_epoch} \
        --logging_steps=1000 \
        --save_steps=5000 \
        --eval_steps=5000 \
        --load_best_model_at_end=True \
        --evaluation_strategy="steps" \
        --save_strategy="steps" \
        --predict_with_generate \
        --model_name_or_path "${pretrained}" \
        --source_lang "${source_key}" \
        --target_lang "${target_key}" \
        --train_file "${train_file}" \
        --validation_file "${dev_file}" \
        --output_dir "${model_dir}" \
        --adafactor \
        --fp16 \
        --fp16_full_eval
done
