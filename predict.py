import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
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

    predicts = []

    if "features" in df_test.columns:
        # Method 1
        # features = df_test["features"].tolist()

        # for text, feature in zip(texts, features):
        #     inputs = (text, feature)
        #     outputs = pipe(inputs)
        #     predicts.append(outputs)
        #     # break
        # probs = np.vstack([[p["score"] for p in predict] for predict in predicts])

        # Method 2, 3
        features = df_test["features"].tolist()
        probs = []
        for text, feature in tqdm(zip(texts, features)):
            # Method 2
            inputs = tokenizer.encode(
                text,
                feature,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            logits = model(inputs.to(torch.device("cuda:0")))[0]  # [:,[0,2]]
            prob = logits.softmax(dim=1)
            probs.append(prob.tolist()[0])

            # Method 3
            # inputs = tokenizer(
            #     text,
            #     feature,
            #     max_length=512,
            #     padding="max_length",
            #     truncation=True,
            #     return_tensors='pt',
            # )
            # logits = model(**inputs.to(torch.device('cuda:0')))[0]   # [:,[0,2]]
            # prob = logits.softmax(dim=1)
            # probs.append(prob.tolist()[0])

        # Method 4: Can't execute
        # features = iter(features)

        # while batch_texts := tuple(islice(texts, args.batch_size)):

        #     batch_features = tuple(islice(features, args.batch_size))
        #     print(batch_features[0])
        #     print(len(batch_texts), len(batch_features))
        #     inputs = [(text, feature) for text, feature in zip(
        #         list(batch_texts), list(batch_features)
        #     )]
        #     outputs = pipe(inputs)
        #     if len(batch_features) == 6:
        #         for item in outputs:
        #             print(item)
        #     predicts.extend(outputs)
        #     # print(len(inputs), len(outputs), len(predicts))
        # probs = np.vstack([[p["score"] for p in predict] for predict in predicts])

    else:
        texts = iter(texts)

        while batch_lines := tuple(islice(texts, args.batch_size)):

            predicts.extend(pipe(list(batch_lines)))

        probs = np.vstack([[p["score"] for p in predict] for predict in predicts])

    np.savetxt(args.out_file, probs, delimiter=" ", fmt="%.5f")


if __name__ == "__main__":
    main()
