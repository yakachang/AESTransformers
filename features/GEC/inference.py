from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "../../AESbackend/t5-small-clean-new"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


def correct(sents, prefix="gec: "):
    # prepare data as the input format of gec model
    sents = [prefix + sent for sent in sents]
    input_ids = tokenizer(
        sents, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    # set max_length as size plus a number in case of insertion edits
    max_length = min(input_ids.size(1) + 10, tokenizer.model_max_length)
    outputs = model.generate(input_ids, max_length=max_length)
    res = [
        tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for output in outputs
    ]
    # TODO: add back truncated text
    return res
