import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LABEL_MAPPER = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}

# Model
PATH_TO_MODEL = (
    "experiments_seq2seq/multi-task/models/"
    + "folds_p1-2/fold_0/"
    + "lr1e-4-b8a1/epoch20-patience5/t5-small-len512-mod"
)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    PATH_TO_MODEL,
    ignore_mismatched_sizes=True,
)

tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error


def add_prefix(trait, text):
    return f"Give a score according to the {trait} of the essay: {text}"


def aes_model(article: str):

    texts = [
        add_prefix(trait, article)
        for trait in ["content", "word choice", "sentence fluency", "conventions"]
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    print(inputs)

    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=2,
        do_sample=False,  # disable sampling to test if batching affects output
    )

    print(output_sequences)

    results = [
        int(pred) if pred and pred.isnumeric() else 0
        for pred in tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    ]

    return results


def main():

    st.title("Automated Essay Scoring")

    article = st.text_area("Please input your essay: ")

    if st.button("Submit"):
        if not article:
            st.write(
                "The content is empty. Please click `Submit` button after you input your essay!"
            )

        else:
            scores = aes_model(article)
            word_count = len(article.strip().split())

            st.write("Scores:")
            st.write(f"\nContent: {scores[0]}")
            st.write(f"\nWord Choice: {scores[1]}")
            st.write(f"\nSentence Fluency: {scores[2]}")
            st.write(f"\nConventions: {scores[3]}")
            st.write(f"Word count: {word_count} words")


if __name__ == "__main__":

    main()
