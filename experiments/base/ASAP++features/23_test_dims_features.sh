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
# for dim in 'language'; do

    setting="lr${lr}-b${batch_size}a${grad_acc}-ada-fp16"
    path_to_model="models/Original/base/sets/${fold_name}/${setting}"
    model_dir="${path_to_model}/${pretrained}-${max_len}-${dim}-mod"
    data_dir="../../../data/ASAP++/Original+features/${fold_name}/${dim}"

    out_dir="${path_to_model}/${pretrained}-${max_len}-${dim}-out"
    mkdir -p "${out_dir}"

    out_file="${out_dir}/test_features.prob"
    if [[ ! -f "${out_file}" ]]; then
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python '../../../predict.py' \
        --checkpoint_file "${model_dir}" \
        --in_file "${data_dir}/test.json" \
        --out_file "${out_file}" \
        --batch_size 64
    fi

    eval_file="${out_dir}/eval.test_features.json"
    if [[ ! -f "${eval_file}" ]]; then
    python '../../../evaluate.py' \
        --min_label "${min_label}" \
        --max_label "${max_label}" \
        --gold_file "${data_dir}/test.json" \
        --prob_file "${out_dir}/test_features.prob" \
        --out_file "${eval_file}"
    fi

    eval_file_cm="${out_dir}/confusion_metrix_features.png"
    if [[ ! -f "${eval_file_cm}" ]]; then
    python '../../../evaluate_cm.py' \
        --set_id "${fold_name}" \
        --gold_file "${data_dir}/test.json" \
        --prob_file "${out_dir}/test_features.prob" \
        --out_file "${eval_file_cm}"
    fi
done
