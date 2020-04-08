##
def n_avg(word_vecs, row=None, lexicon=None):
    """Takes the average of the vectors, word_vecs

    Parameters:
        word_vecs (list:vectors):  A list of the word vectors

    Returns:
        (vector): The resulting averaged vector

    """
    return sum(word_vecs)/len(word_vecs)


def lexicon_avg(word_vecs, row, lexicon):
    """Takes the weighted average of the vectors.
       Weighted according to the lexicon score with base weight = 1/len(word_vecs)

        Parameters:
            word_vecs (list:vectors):  A list of the word vectors
            row (pandas Series): The row in the dataframe corresponding with the tweet of the given word_vecs
            lexicon (Lexiocon): The lexicon used to get the weights of the words

        Returns:
            (vector): The resulting weighted averaged vector

        """
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
    """Takes the weighted average of the vectors.
       Weighted according to the lexicon score with base weight = 1/len(word_vecs)

        Parameters:
            word_vecs (list:vectors):  A list of the word vectors
            row (pandas Series): The row in the dataframe corresponding with the tweet of the given word_vecs
            lexicons (list:Lexiocon): The lexicon used to get the weights of the words

        Returns:
            (vector): The resulting weighted averaged vector

        """
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
