import re
import pandas as pd


def collect_dataset(input_file):

    df_ori = pd.read_csv(input_file, sep="\t", encoding="ISO-8859-1")
    df = df_ori[["essay"]]
    df.columns = ["text"]

    return df


input_file = "../../data/ASAP/kaggle-asap-aes/training_set_rel3.tsv"
dataset = collect_dataset(input_file)
texts = dataset["text"].tolist()

special_tokens = set()

for text in texts:
    tokens = text.split()
    for token in tokens:
        if token[0] == "@" and len(token) > 1:
            token = re.match(r"@[A-Z]+[0-9]+", token).group(0)
            special_tokens.add(token)

print("Special Tokens:")
print(special_tokens)
