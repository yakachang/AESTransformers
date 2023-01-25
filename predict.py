import argparse
import numpy as np
import pandas as pd

from itertools import islice
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_file, ignore_mismatched_sizes=True
    )
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_all_scores=True,
        batch_size=args.batch_size,
        device=0,
    )

    df_test = pd.read_json(args.in_file, lines=True)
    texts = df_test["text"].tolist()
    texts = iter(texts)

    predicts = []

    while batch_lines := tuple(islice(texts, args.batch_size)):

        predicts.extend(pipe(list(batch_lines)))

    probs = np.vstack([[p["score"] for p in predict] for predict in predicts])
    np.savetxt(args.out_file, probs, delimiter=" ", fmt="%.5f")


if __name__ == "__main__":
    main()
