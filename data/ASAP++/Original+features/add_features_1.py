import pandas as pd

dim_mapper = {
    "set1": ["word_choice", "sentence_fluency", "conventions"],
    "set2": ["language"],
}

CEFR_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]


def add_word_info(df):
    def concat_info(item):
        info_list = []
        for k, v in item["vocab"].items():
            if k == "vocab_level_dist":
                level_info = []
                for level in CEFR_levels:
                    level_info.append(f"{level}: {round(v.get(level, 0), 4)}")
                info_list.append(f"{k}:({', '.join(level_info)})")
            else:
                info_list.append(f"{k}:{v}")
        return "; ".join(info_list)

    df["features"] = df["feedback"].apply(lambda x: concat_info(x))
    # print(df["features"].tolist()[0])

    return df


def add_sent_info(df):
    def concat_info(item):
        info_list = []
        for k, v in item["sent"].items():
            info_list.append(f"{k}:{round(v, 2)}")
        return "; ".join(info_list)

    df["features"] = df["feedback"].apply(lambda x: concat_info(x))
    # print(df["features"].tolist()[0])

    return df


def add_gec_info(df):
    def concat_info(item):
        info_list = []
        types = [
            "spelling",
            "usage_error_num",
            "form_usage",
            "prep_usage",
            "punc_usage",
            "conj_usage",
        ]
        for type in types:
            correct, mistake = item["grammar"][type].values()
            if (correct + mistake) > 0:
                info_list.append(
                    f"{type.split('_')[0]}_error:{round(mistake / (correct + mistake), 2)}"
                )
        return "; ".join(info_list)

    df["features"] = df["feedback"].apply(lambda x: concat_info(x))
    # print(df["features"].tolist()[0])

    return df


def add_features(dim, df):

    if dim == "word_choice":
        df = add_word_info(df)
        return df
    elif dim == "sentence_fluency":
        df = add_sent_info(df)
        return df
    elif dim == "conventions":
        df = add_gec_info(df)
        return df
    elif dim == "language":
        df = add_gec_info(df)
        return df


def export_file(set_id, dim, datatype, df):

    df.to_json(
        f"{set_id}/{dim}/{datatype}.json",
        orient="records",
        lines=True,
    )


def main():

    df_features = pd.read_json("../../ASAP/features/ASAP+features.jsonl", lines=True)
    df_features = df_features[["essay_id", "feedback"]]

    for set_id in ["set1", "set2"]:
        for dim in dim_mapper[set_id]:
            for datatype in ["train", "dev"]:
                df = pd.read_json(
                    f"../Original/{set_id}/{dim}/{datatype}.json", lines=True
                )
                df = df.merge(df_features, how="inner", on="essay_id")
                df = add_features(dim, df)
                df = df[["essay_id", "prompt_id", "label", "text", "features"]]
                export_file(set_id, dim, datatype, df)
                # print(df.head())
                # break
            # break
        # break


if __name__ == "__main__":
    main()
