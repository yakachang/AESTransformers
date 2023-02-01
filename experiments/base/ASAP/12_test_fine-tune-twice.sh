#!/bin/bash

fold_name="fold_0"
pretrained="bert-base-cased"
max_len=512
lr=2e-5
batch_size=8
grad_acc=1
path_to_model="models/Original/base/sets/${fold_name}/lr${lr}-b${batch_size}a${grad_acc}"
model_dir="${path_to_model}/${pretrained}-${max_len}-mod"
data_dir="../../../data/ASAP/folds/Test-ori/${fold_name}"

out_dir="${path_to_model}/${pretrained}-${max_len}-b${batch_size}a${grad_acc}-out"
mkdir -p "${out_dir}/probs"
mkdir -p "${out_dir}/evals"
mkdir -p "${out_dir}/cms"


# For each prompt
for id in {1..8}
do
  out_file="${out_dir}/probs/test_${id}.prob"
  if [[ ! -f "${out_file}" ]]; then
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python '../../../predict.py' \
    --checkpoint_file "${model_dir}" \
    --in_file "${data_dir}/test_${id}.jsonl" \
    --out_file "${out_file}" \
    --batch_size 64
  fi

  eval_file="${out_dir}/evals/eval.test_${id}.json"
  if [[ ! -f "${eval_file}" ]]; then
  python '../../../evaluate.py' \
    --prompt_id "${id}" \
    --gold_file "${data_dir}/test_${id}.jsonl" \
    --prob_file "${out_dir}/probs/test_${id}.prob" \
    --out_file "${eval_file}" \
    --rescale_score
  fi

  eval_file_cm="${out_dir}/cms/confusion_metrix_${id}.png"
  if [[ ! -f "${eval_file_cm}" ]]; then
    python '../../../evaluate_cm.py' \
      --prompt_id "${id}" \
      --gold_file "${data_dir}/test_${id}.jsonl" \
      --prob_file "${out_dir}/probs/test_${id}.prob" \
      --out_file "${eval_file_cm}" \
      --rescale_score
  fi
done
