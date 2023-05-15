import torch
import argparse

import numpy as np
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


def add_prefix(text):

    trait, text = text.split(":", 1)
    trait = " ".join(trait.split("_"))

    return f"Give a score according to the {trait} of the essay: {text}"


def main():
    args = build_args()

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_file)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint_file,
        ignore_mismatched_sizes=True,
    ).to(device)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    df_test = pd.read_json(args.in_file, lines=True)
    texts = df_test["text"].tolist()
    texts = [add_prefix(text) for text in texts]

    predicts = []

    texts = iter(texts)

    while batch_lines := tuple(islice(texts, args.batch_size)):

        inputs = tokenizer(
            list(batch_lines),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=2,
            do_sample=False,  # disable sampling to test if batching affects output
        ).to(device)
        predicts.extend(
            [
                int(pred)
                for pred in tokenizer.batch_decode(
                    output_sequences, skip_special_tokens=True
                )
            ]
        )

    probs = np.vstack([p for p in predicts])

    np.savetxt(args.out_file, probs, delimiter=" ", fmt="%i")


if __name__ == "__main__":
    main()
