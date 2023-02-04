#!/bin/bash

fold_name="set1"
# Set 1: "label", "content", "organization", "word_choice", "sentence_fluency", "conventions"
# Set 2: "label", "content", "prompt_adherence", "language", "narrativity"
# dim="content"
min_label=1
max_label=6

pretrained="bert-base-cased"
max_len=512
lr=2e-5
batch_size=8
grad_acc=1

# 'content' 'organization' 'word_choice' 'sentence_fluency' 'conventions'
# 'content' 'prompt_adherence' 'language' 'narrativity'
for dim in 'word_choice' 'sentence_fluency' 'conventions'; do

    setting="lr${lr}-b${batch_size}a${grad_acc}-ada-fp16"
    path_to_model="models/Original/base/sets/${fold_name}/${setting}"
    model_dir="${path_to_model}/${pretrained}-${max_len}-${dim}-mod"
    data_dir="../../../data/ASAP++/Original+features/${fold_name}/${dim}"

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
        --model_name_or_path "${pretrained}" \
        --train_file "${data_dir}/train.json" \
        --validation_file "${data_dir}/dev.json" \
        --overwrite_output_dir \
        --output_dir "${model_dir}" \
        --min_label "${min_label}" \
        --max_label "${max_label}" \
        --dataset_source "dim"

done
