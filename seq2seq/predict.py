# import torch
import argparse

# import numpy as np
import pandas as pd

# from tqdm import tqdm
from itertools import islice
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_file)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint_file, ignore_mismatched_sizes=True
    )

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    df_test = pd.read_json(args.in_file, lines=True)
    texts = df_test["text"].tolist()

    # predicts = []

    texts = iter(texts)

    while batch_lines := tuple(islice(texts, 10)):

        inputs = tokenizer(
            list(batch_lines),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1,
            do_sample=False,  # disable sampling to test if batching affects output
        )
        print(output_sequences)
        print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
        break

    # while batch_lines := tuple(islice(texts, args.batch_size)):

    #     predicts.extend(pipe(list(batch_lines)))

    # probs = np.vstack([[p["score"] for p in predict] for predict in predicts])

    # np.savetxt(args.out_file, probs, delimiter=" ", fmt="%.5f")


if __name__ == "__main__":
    main()
