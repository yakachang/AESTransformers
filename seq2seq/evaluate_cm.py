import argparse
import jsonlines
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


def get_labels(prompt_id):

    return [
        i
        for i in range(
            bound_mapper[prompt_id]["min"], bound_mapper[prompt_id]["max"] + 1
        )
    ]


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
    parser.add_argument("--prompt_id", type=int, default=None)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    gold_labels = load_gold_labels(args.gold_file, args.prompt_id)
    pred_labels = load_pred_labels(args.gold_file, args.prob_file, args.prompt_id)

    assert len(gold_labels) == len(pred_labels)

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
