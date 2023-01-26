import argparse
import json
import jsonlines
import numpy as np

from processors import score_converter
from sklearn.metrics import mean_squared_error, cohen_kappa_score


def load_pred_labels(filename, min_label, max_label):
    probs = np.loadtxt(filename, dtype=np.float64)
    pred_labels = np.argmax(probs, axis=1)
    i2label = {
        i: label for i, label in enumerate([i for i in range(min_label, max_label + 1)])
    }
    return [i2label[pred] for pred in pred_labels]


def load_gold_labels(filename, rescale_score=False):

    if rescale_score:
        scores = [line["label"] for line in jsonlines.open(filename)]
        prompt_ids = [line["prompt_id"] for line in jsonlines.open(filename)]
        return [
            score_converter(prompt_id, score)
            for prompt_id, score in zip(prompt_ids, scores)
        ]

    else:
        return [line["label"] for line in jsonlines.open(filename)]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_id", type=int, default=None)
    parser.add_argument("--min_label", type=int, default=0)
    parser.add_argument("--max_label", type=int, default=60)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--rescale_score", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    gold_labels = load_gold_labels(args.gold_file, args.rescale_score)
    pred_labels = load_pred_labels(args.prob_file, args.min_label, args.max_label)

    mse = mean_squared_error(gold_labels, pred_labels)
    qwk = cohen_kappa_score(gold_labels, pred_labels, weights="quadratic")

    results = {
        "Prompt_ID": args.prompt_id,
        "mse": mse,
        "qwk": qwk,
    }

    print(results)

    with open(args.out_file, "w") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    main()
