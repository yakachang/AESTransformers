#!/bin/bash

set -ex

lr=1e-4
max_len=512
batch_size=8
grad_acc=1
max_epoch=20
patience=5
pretrained="t5-small"

for fold_id in "fold_0" "fold_1" "fold_2" "fold_3" "fold_4";
do
    fold_path="folds_p3-6/${fold_id}"
    data_dir="../../data/ASAP++/Multi-Task/${fold_path}"
    setting="epoch${max_epoch}-patience${patience}"
    base_path="models/${fold_path}/lr${lr}-b${batch_size}a${grad_acc}/${setting}"
    model_dir="${base_path}/${pretrained}-len${max_len}-mod"

    unset -v latest

    out_dir="${base_path}/${pretrained}-len${max_len}-out"
    mkdir -p "${out_dir}/probs"
    mkdir -p "${out_dir}/evals"
    mkdir -p "${out_dir}/cms"

    for trait in "content" "prompt_adherence" "language" "narrativity";
    do
        out_file="${out_dir}/probs/test_${trait}.prob"
        if [[ ! -f "${out_file}" ]]; then
        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python '../../seq2seq/predict.py' \
            --checkpoint_file "${model_dir}" \
            --in_file "${data_dir}/test/test_${trait}.jsonl" \
            --out_file "${out_file}" \
            --batch_size 64
        fi

        eval_file="${out_dir}/evals/eval.test_${trait}.json"
        if [[ ! -f "${eval_file}" ]]; then
        python '../../seq2seq/evaluate.py' \
            --gold_file "${data_dir}/test/test_${trait}.jsonl" \
            --prob_file "${out_dir}/probs/test_${trait}.prob" \
            --out_file "${eval_file}" \
            --prompt_ids "3,4,5,6"
        fi

        for id in {3..6};
        do
            eval_file_cm="${out_dir}/cms/cm_${trait}_${id}.png"
            if [[ ! -f "${eval_file_cm}" ]]; then
            python '../../seq2seq/evaluate_cm.py' \
                --prompt_id "${id}" \
                --gold_file "${data_dir}/test/test_${trait}.jsonl" \
                --prob_file "${out_dir}/probs/test_${trait}.prob" \
                --out_file "${eval_file_cm}"
            fi
        done
    done
done
