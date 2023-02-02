def get_counts(doc, pos_list=["VERB", "ADJ", "NOUN"]):
    count = 0

    for sent in doc.sents:
        for token in sent:
            if token.pos_ in pos_list:
                count += 1

    return count
