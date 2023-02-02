import pandas as pd
import dimensional_analyzer


def collect_dataset(input_file):

    df_ori = pd.read_csv(input_file, sep="\t", encoding="ISO-8859-1")
    df = df_ori[["essay_id", "essay_set", "essay", "domain1_score"]]
    df.columns = ["essay_id", "prompt_id", "text", "label"]

    return df


def main():

    input_file = "../../data/ASAP/kaggle-asap-aes/training_set_rel3.tsv"
    dataset = collect_dataset(input_file)

    dataset["feedback"] = dataset["text"].apply(
        lambda text: dimensional_analyzer.analyze_text(text)
    )

    dataset.to_json(
        "../data/ASAP/features/ASAP+features.jsonl",
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
