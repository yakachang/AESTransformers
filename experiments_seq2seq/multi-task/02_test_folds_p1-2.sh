#!/bin/bash

fold_name="folds_p1-2"
fold_id="fold_0"
trait="content"

# pretrained="google/flan-t5"
# max_len=512
# lr=2e-5
# batch_size=8
# grad_acc=1

data_dir="../../data/ASAP++/Multi-Task/${fold_name}/${fold_id}"

# path_to_model="models/Original/base/sets/${fold_name}/lr${lr}-b${batch_size}a${grad_acc}"
# model_dir="${path_to_model}/${pretrained}-${max_len}-${trait}-mod"
model_dir="test-mod"

# out_dir="${path_to_model}/${pretrained}-${max_len}-${trait}-out"
out_dir="test-out"
mkdir -p "${out_dir}"

out_file="${out_dir}/test.prob"
if [[ ! -f "${out_file}" ]]; then
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python '../../seq2seq/predict.py' \
    --checkpoint_file "${model_dir}" \
    --in_file "${data_dir}/test/test_${trait}.jsonl" \
    --out_file "${out_file}" \
    --batch_size 64
fi

# eval_file="${out_dir}/eval.test.json"
# if [[ ! -f "${eval_file}" ]]; then
# python '../../../evaluate.py' \
#     --gold_file "${data_dir}/test.json" \
#     --prob_file "${out_dir}/test.prob" \
#     --out_file "${eval_file}"
# fi

# eval_file_cm="${out_dir}/confusion_metrix.png"
# if [[ ! -f "${eval_file_cm}" ]]; then
# python '../../../evaluate_cm.py' \
#     --set_id "${fold_name}" \
#     --gold_file "${data_dir}/test.json" \
#     --prob_file "${out_dir}/test.prob" \
#     --out_file "${eval_file_cm}"
# fi
