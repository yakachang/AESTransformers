import argparse
import json
import jsonlines
import numpy as np

from sklearn.metrics import mean_squared_error, cohen_kappa_score


bound_mapper = {
    1: {"min": 1, "max": 6},
    2: {"min": 1, "max": 6},
    3: {"min": 0, "max": 3},
    4: {"min": 0, "max": 3},
    5: {"min": 0, "max": 4},
    6: {"min": 0, "max": 4},
    7: {"min": 0, "max": 3},
    8: {"min": 1, "max": 6},
}


def load_pred_labels(gold_file, filename, prompt_id):

    prompt_ids = [line["prompt_id"] for line in jsonlines.open(gold_file)]

    pred_file = open(filename, "r")
    probs = pred_file.read().replace("\n", " ").split()

    pred_labels = [
        int(prob) for idx, prob in zip(prompt_ids, probs) if idx == prompt_id
    ]

    pred_file.close()

    return pred_labels


def load_gold_labels(filename, prompt_id):

    prompt_ids = [line["prompt_id"] for line in jsonlines.open(filename)]
    labels = [line["label"] for line in jsonlines.open(filename)]

    filtered_labels = [
        label for idx, label in zip(prompt_ids, labels) if idx == prompt_id
    ]

    return filtered_labels


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--prompt_ids", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    results = {}
    qwk_all = []

    prompt_ids = [int(prompt_id) for prompt_id in args.prompt_ids.split(",")]

    for prompt_id in prompt_ids:
        gold_labels = load_gold_labels(args.gold_file, prompt_id)
        pred_labels = load_pred_labels(args.gold_file, args.prob_file, prompt_id)

        assert len(gold_labels) == len(pred_labels)

        mse = mean_squared_error(gold_labels, pred_labels)
        qwk = cohen_kappa_score(gold_labels, pred_labels, weights="quadratic")

        results[f"prompt_id_{prompt_id}"] = {
            "mse": np.round(mse, 3),
            "qwk": np.round(qwk, 3),
        }

        qwk_all.append(np.round(qwk, 3))

    results["qwk_avg"] = np.round(sum(qwk_all) / len(qwk_all), 3)
    print(results)

    with open(args.out_file, "w") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    main()
