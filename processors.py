def score_converter(prompt_id, score):

    if prompt_id == 1:
        return score * 6 - 12
    elif prompt_id == 2:
        return (score - 1) * 12
    elif prompt_id in [3, 4]:
        return score * 20
    elif prompt_id in [5, 6]:
        return score * 15
    elif prompt_id == 7:
        return score * 2
    else:
        return score
