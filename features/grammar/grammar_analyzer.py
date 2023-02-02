from . import utils


def get_grammar_analysis(grammar_error_types, doc, STOP_WORDS):

    word_count = utils.get_word_count(doc)
    spell_error_num = grammar_error_types.get("Spelling", 0)

    vocabs = utils.get_vocabs(doc, STOP_WORDS)
    usage_error_num = utils.get_usage_error_num(grammar_error_types)

    form_error_num = grammar_error_types.get("Form", 0)
    form_num = utils.get_counts(doc)

    prep_error_num = grammar_error_types.get("Preposition", 0)
    prep_num = utils.get_counts(doc, pos_list=["ADV", "PART"])

    punc_error_num = grammar_error_types.get("Punctuation", 0)
    punc_num = utils.get_counts(doc, pos_list=["PUNCT"])

    conj_error_num = grammar_error_types.get("Conjunction", 0)
    conj_num = utils.get_counts(doc, pos_list=["CONJ"])

    return {
        "spelling": {
            "correct": word_count - spell_error_num,
            "mistake": spell_error_num,
        },
        "usage_error_num": {
            "correct": len(vocabs) - usage_error_num,
            "mistake": usage_error_num,
        },
        "form_usage": {"correct": form_num - form_error_num, "mistake": form_error_num},
        "prep_usage": {"correct": prep_num - prep_error_num, "mistake": prep_error_num},
        "punc_usage": {"correct": punc_num - punc_error_num, "mistake": punc_error_num},
        "conj_usage": {"correct": conj_num - conj_error_num, "mistake": conj_error_num},
    }
