import os
import pandas as pd


def determine_group(item):
    if item > 0.95:
        return 1
    elif item > 0.85:
        return 0.9
    elif item > 0.75:
        return 0.8
    elif item > 0.65:
        return 0.7
    elif item > 0.55:
        return 0.6
    elif item > 0.45:
        return 0.5
    elif item > 0.35:
        return 0.4
    elif item > 0.25:
        return 0.3
    elif item > 0.15:
        return 0.2
    elif item > 0.05:
        return 0.1
    return 0.0


def main():

    datatypes = ["train", "dev"]
    folds = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

    for fold_name in folds:
        print(f"fold name: {fold_name}")
        path_to_input_folder = f"../data/ASAP/folds/Min-Max-Scaling/{fold_name}"
        path_to_output_folder = (
            f"../data/ASAP/folds/Min-Max-Scaling-balanced/{fold_name}"
        )

        for datatype in datatypes:
            print(f"datatype: {datatype}")

            df = pd.read_json(
                os.path.join(path_to_input_folder, f"{datatype}.json"), lines=True
            )

            df["label"] = df["label"].apply(lambda x: determine_group(x))

            df_balance = df.groupby("label")
            df_balance = df_balance.apply(
                lambda x: x.sample(df_balance.size().max(), replace=True).reset_index(
                    drop=True
                )
            )
            df_balance = df_balance[["essay_id", "prompt_id", "label", "score", "text"]]

            # print(df_balance.head())
            # print(df_balance["label"].value_counts())

            df_balance.to_json(
                os.path.join(path_to_output_folder, f"{datatype}.json"),
                orient="records",
                lines=True,
            )


if __name__ == "__main__":
    main()
