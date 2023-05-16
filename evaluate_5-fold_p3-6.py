import os
import json
import numpy as np


trait_prompt = {
    "content": [3, 4, 5, 6],
    "organization": [3, 4, 5, 6],
    "word_choice": [3, 4, 5, 6],
    "sentence_fluency": [3, 4, 5, 6],
    "conventions": [3, 4, 5, 6],
}


def main():
    lr = "lr1e-4"
    batch = "b8a1"
    max_len = 512

    path_to_folder = "experiments_seq2seq/multi-task/models/folds_p3-6"

    # "t5-small" "t5-base"
    pretrained = "t5-small"

    results = {
        "Experiment Setting": {
            "PLM": pretrained,
            "lr": lr,
            "batch": batch,
            "max_len": max_len,
        }
    }

    for trait in trait_prompt.keys():
        results[trait] = {}
        for prompt_id in [1, 2]:
            mse_scores = []
            qwk_scores = []
            for fold in [f"fold_{idx}" for idx in range(5)]:
                with open(
                    os.path.join(
                        path_to_folder,
                        f"{fold}/{lr}-{batch}",
                        f"epoch20-patience5/{pretrained}-len{max_len}-out",
                        f"evals/eval.test_{trait}.json",
                    )
                ) as json_file:
                    data = json.load(json_file)
                    mse_scores.append(data[f"prompt_id_{prompt_id}"]["mse"])
                    qwk_scores.append(data[f"prompt_id_{prompt_id}"]["qwk"])
                    # break
            results[trait][f"prompt_id_{prompt_id}"] = {
                "mse": np.round(sum(mse_scores) / len(mse_scores), 3),
                "qwk": np.round(sum(qwk_scores) / len(qwk_scores), 3),
            }
        # break
        qwk_scores = [
            results[trait][f"prompt_id_{idx}"]["qwk"] for idx in trait_prompt[trait]
        ]
        results[trait]["qwk_avg"] = np.round(sum(qwk_scores) / len(qwk_scores), 3)
    print(results)

    with open(
        os.path.join(
            f"{path_to_folder}/fold_avg",
            f"eval.test.{pretrained}.{lr}-{batch}-len{max_len}.json",
        ),
        "w",
    ) as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    main()
