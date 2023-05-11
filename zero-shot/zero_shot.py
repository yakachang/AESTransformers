import jsonlines

from itertools import islice
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


pretrained_model = "bigscience/mt0-base"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)


def read_data(filename):

    texts = [line["text"] for line in jsonlines.open(filename)]
    labels = [line["label"] for line in jsonlines.open(filename)]

    return texts, labels


def add_prefix(prefix, texts):

    return [f"{prefix}: {text}" for text in texts]


def main():

    prefix = "Get content score in range 1-6"

    path_to_data = "../AESLightning/data/ASAP++/folds/fold_0/content/test.jsonl"
    texts, labels = read_data(path_to_data)
    texts = iter(texts)

    while batch_lines := tuple(islice(texts, 2)):
        inputs = add_prefix(prefix, batch_lines)
        # for item in inputs:
        #     print(item)
        inputs = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        outputs = model.generate(**inputs)
        print(tokenizer.decode(outputs[0]))
        break


if __name__ == "__main__":
    main()
