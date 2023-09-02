#!/bin/bash

set -ex

lr=1e-4
max_len=512
batch_size=8
grad_acc=1
max_epoch=20
patience=5
pretrained="t5-small"

fold_id="fold_0"

data_dir="../../data"
fold_path="folds_p8/${fold_id}"
setting="epoch${max_epoch}-patience${patience}"
base_path="models/${fold_path}/lr${lr}-b${batch_size}a${grad_acc}/${setting}"
model_dir="${base_path}/${pretrained}-len${max_len}-mod"

unset -v latest

out_dir="${pretrained}-len${max_len}-out"
mkdir -p "${out_dir}"

for trait in "voice" "something";
do
    out_file="${out_dir}/test_${trait}.prob"
    if [[ ! -f "${out_file}" ]]; then
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python '../../seq2seq/predict.py' \
        --checkpoint_file "${model_dir}" \
        --in_file "${data_dir}/human-study/${trait}/test.jsonl" \
        --out_file "${out_file}" \
        --trait "${trait}" \
        --batch_size 64
    fi
done
