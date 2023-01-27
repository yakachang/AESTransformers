#!/bin/bash

fold_name="set1"
min_label=1
max_label=6
# fold_name="set2"
# min_label=0
# max_label=4

pretrained="bert-base-cased"
max_len=512
lr=2e-5
batch_size=8
grad_acc=1

setting="lr${lr}-b${batch_size}a${grad_acc}-ada-fp16"
path_to_model="models/Original/base/sets/${fold_name}/${setting}"
model_dir="${path_to_model}/${pretrained}-${max_len}-mix-mod"
data_dir="../../../data/ASAP++/Original/${fold_name}/mix-dim"

out_dir="${path_to_model}/${pretrained}-${max_len}-mix-out"
mkdir -p "${out_dir}/probs"
mkdir -p "${out_dir}/evals"
mkdir -p "${out_dir}/cms"

# Set 1: 'content' 'organization' 'word_choice' 'sentence_fluency' 'conventions'
# Set 2: 'content' 'prompt_adherence' 'language' 'narrativity'
for dim in 'content' 'organization' 'word_choice' 'sentence_fluency' 'conventions'; do

    prob_file="${out_dir}/probs/test_${dim}.prob"
    if [[ ! -f "${prob_file}" ]]; then
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python '../../../predict.py' \
        --checkpoint_file "${model_dir}" \
        --in_file "${data_dir}/test_${dim}.json" \
        --out_file "${prob_file}" \
        --batch_size 64
    fi

    eval_file="${out_dir}/evals/eval_${dim}.json"
    if [[ ! -f "${eval_file}" ]]; then
    python '../../../evaluate.py' \
        --min_label "${min_label}" \
        --max_label "${max_label}" \
        --gold_file "${data_dir}/test_${dim}.json" \
        --prob_file "${prob_file}" \
        --out_file "${eval_file}"
    fi

    cm_file="${out_dir}/cms/confusion_metrix_${dim}.png"
    if [[ ! -f "${cm_file}" ]]; then
    python '../../../evaluate_cm.py' \
        --set_id "${fold_name}" \
        --gold_file "${data_dir}/test_${dim}.json" \
        --prob_file "${prob_file}" \
        --out_file "${cm_file}"
    fi
done
