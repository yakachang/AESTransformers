import argparse
import math
import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

bound_mapper = {
    1: {"min": 2, "max": 12},
    2: {"min": 1, "max": 6},
    3: {"min": 0, "max": 3},
    4: {"min": 0, "max": 3},
    5: {"min": 0, "max": 4},
    6: {"min": 0, "max": 4},
    7: {"min": 2, "max": 24},
    8: {"min": 10, "max": 60},
}


def get_labels(prompt_id):

    if prompt_id == 1:
        return [i for i in range(2, 12 + 1)]
    elif prompt_id == 2:
        return [i for i in range(1, 6 + 1)]
    elif prompt_id == 3:
        return [i for i in range(0, 3 + 1)]
    elif prompt_id == 4:
        return [i for i in range(0, 3 + 1)]
    elif prompt_id == 5:
        return [i for i in range(0, 4 + 1)]
    elif prompt_id == 6:
        return [i for i in range(0, 4 + 1)]
    elif prompt_id == 7:
        return [i for i in range(0, 30 + 1)]
    elif prompt_id == 8:
        return [i for i in range(0, 60 + 1)]


def score_converter(prompt_id, score):

    assert score is not None

    if prompt_id == 1:
        return math.floor((score + 12) / 6)
    elif prompt_id == 2:
        return math.floor(score / 12 + 1)
    elif prompt_id == 3:
        return math.floor(score / 20)
    elif prompt_id == 4:
        return math.floor(score / 20)
    elif prompt_id == 5:
        return math.floor(score / 15)
    elif prompt_id == 6:
        return math.floor(score / 15)
    elif prompt_id == 7:
        return math.floor(score / 2)
    else:
        return score


def load_pred_labels(filename, prompt_id=None, set_id=None):
    probs = np.loadtxt(filename, dtype=np.float64)
    pred_labels = np.argmax(probs, axis=1)
    if prompt_id:
        i2label = {i: label for i, label in enumerate([i for i in range(0, 60 + 1)])}
        return [score_converter(prompt_id, i2label[pred]) for pred in pred_labels]
    elif set_id:
        if set_id == "set1":
            i2label = {i: label for i, label in enumerate([i for i in range(1, 6 + 1)])}
        elif set_id == "set2":
            i2label = {i: label for i, label in enumerate([i for i in range(0, 4 + 1)])}
        return [i2label[pred] for pred in pred_labels]


def load_gold_labels(filename):

    return [line["label"] for line in jsonlines.open(filename)]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_id", type=int, default=None)
    parser.add_argument("--set_id", type=str, default=None)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--rescale_score", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    gold_labels = load_gold_labels(args.gold_file)
    pred_labels = load_pred_labels(args.prob_file, args.prompt_id, args.set_id)

    cm = confusion_matrix(gold_labels, pred_labels, labels=get_labels(args.prompt_id))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=get_labels(args.prompt_id)
    )

    disp.plot()
    plt.title(f"Confusion Matrix for Prompt {args.prompt_id}")
    plt.show()
    plt.savefig(args.out_file)


if __name__ == "__main__":
    main()
