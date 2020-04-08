##
def n_avg(word_vecs, row=None, lexicon=None):
    return sum(word_vecs)/len(word_vecs)


def lexicon_avg(word_vecs, row, lexicon):
    weights = []
    words = row["Words"]
    for i, word_vec in enumerate(word_vecs):
        word = words[i]
        w = 1/len(word_vecs)
        if word in lexicon.wordlex.keys():
            w = lexicon.wordlex[word]
        weights.append(abs(w))
    return sum(word_vec * w for word_vec, w in zip(word_vecs, weights)) / sum(weights)


def all_lexicon_avg(word_vecs, row, lexicons):
    weights = []
    words = row["Words"]
    for i, word_vec in enumerate(word_vecs):
        word = words[i]
        w = 1/len(word_vecs)
        for lexicon in lexicons:
            if word in lexicon.wordlex.keys():
                w += lexicon.wordlex[word]
        weights.append(w)
    return sum(word_vec * w for word_vec, w in zip(word_vecs, weights)) / sum(weights)
