import pandas as pd


dim_mapper = {
    1: [
        "content",
        "organization",
        "word_choice",
        "sentence_fluency",
        "conventions",
    ],
    2: ["content", "prompt_adherence", "language", "narrativity"],
}


def main():

    for set_id in [1, 2]:
        for datatype in ["train", "dev", "test"]:
            df_list = []
            for dim in dim_mapper[set_id]:
                df = pd.read_json(
                    f"../data/ASAP++/Original/set{set_id}/{dim}/{datatype}.json",
                    lines=True,
                )
                df["text"] = df["text"].apply(lambda x: f"{dim}: {x}")
                if datatype == "test":
                    df.to_json(
                        f"../data/ASAP++/Original/set{set_id}/mix-dim/test_{dim}.json",
                        orient="records",
                        lines=True,
                    )
                else:
                    df_list.append(df)

            if datatype != "test":

                df_all = pd.concat(df_list)

                df_all.sample(frac=1, random_state=42).to_json(
                    f"../data/ASAP++/Original/set{set_id}/mix-dim/{datatype}.json",
                    orient="records",
                    lines=True,
                )


if __name__ == "__main__":

    main()
