#!/usr/bin/env python

# Source: https://github.com/nusnlp/nea
# Script to pre-process ASAP dataset (training_set_rel3.tsv) based on the essay IDs
"""
Basic Usage
[Command]
python preprocess_asap_dim.py \
    --output_folder_name "../data/ASAP++/Original"
"""

import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


seed = 42
path_to_dim = "../../data/ASAP/ASAP++/Scores"
set_id_mapper = {1: [1, 2], 2: [3, 4, 5, 6]}

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder_name", type=str, required=True)
parser.add_argument(
    "--balance_data",
    action="store_true",
    help="Determine balance label in each prompt or not",
)
args = parser.parse_args()

dim_mapper = {
    1: [
        "label",
        "content",
        "organization",
        "word_choice",
        "sentence_fluency",
        "conventions",
    ],
    2: ["label", "content", "prompt_adherence", "language", "narrativity"],
}


def collect_dataset(input_file):

    df_ori = pd.read_csv(input_file, sep="\t", encoding="ISO-8859-1")
    df = df_ori[["essay_id", "essay_set", "essay", "domain1_score"]]
    df.columns = ["essay_id", "prompt_id", "text", "label"]

    df_all = pd.DataFrame()

    for i in range(1, 6 + 1):

        df_dim = pd.read_csv(os.path.join(path_to_dim, f"Prompt-{i}.csv"))
        # Change column name
        if i == 1:
            df_dim.rename(columns={"EssayID": "essay_id"}, inplace=True)
        else:
            df_dim.rename(columns={"Essay ID": "essay_id"}, inplace=True)
        # Remove not existed data
        if i == 4:
            df_dim.drop(
                df_dim.loc[df_dim["essay_id"].isin([10535, 10536])].index, inplace=True
            )

        df_new = pd.DataFrame()

        for item in df_dim.iterrows():
            essay_id = item[1]["essay_id"]

            df_new = pd.concat(
                [df_new, df.loc[df["essay_id"] == essay_id]], ignore_index=True
            )

        for item1, item2 in zip(df_new.iterrows(), df_dim.iterrows()):
            if item1[1]["essay_id"] != item2[1]["essay_id"]:
                print(item1[1]["essay_id"], item2[1]["essay_id"])

        columns = [col for col in df_dim.columns][1:]

        for col in columns:
            df_new["_".join(col.split(" ")).lower()] = [
                int(s) for s in df_dim[col].tolist()
            ]

        df_all = pd.concat([df_all, df_new], ignore_index=True)

    return df_all


def generate_data(dataset, set_id):

    df_out = dataset.loc[dataset["prompt_id"].isin(set_id_mapper[set_id])]
    df_out = df_out.dropna(axis="columns")

    return df_out


def convert_to_jsonlines(df, set_id):

    # Shuffle
    df = df.sample(frac=1, random_state=seed)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_test, df_dev = train_test_split(df_test, test_size=0.5, random_state=seed)

    filenames = ["train", "dev", "test"]
    df_lists = [df_train, df_dev, df_test]

    for data, filename in zip(df_lists, filenames):
        output_fname = f"{args.output_folder_name}/{filename}.jsonl"
        if os.path.exists(output_fname):
            raise FileExistsError(f"The file {output_fname} is already existed.")
        else:
            for dim in dim_mapper[set_id]:
                df_tmp = data[["essay_id", "prompt_id", "text", dim]]

                if dim == "label":
                    dim = "holistic"
                else:
                    df_tmp.rename(columns={dim: "label"}, inplace=True)
                    df_tmp["label"] = df_tmp["label"].astype("int64")

                df_tmp.to_json(
                    f"{args.output_folder_name}/set{set_id}/{dim}/{filename}.json",
                    orient="records",
                    lines=True,
                )


if __name__ == "__main__":

    input_file = "../../data/ASAP/kaggle-asap-aes/training_set_rel3.tsv"
    dataset = collect_dataset(input_file)

    for set_id in [1, 2]:

        df = generate_data(dataset, set_id)

        convert_to_jsonlines(df, set_id)

        # break
