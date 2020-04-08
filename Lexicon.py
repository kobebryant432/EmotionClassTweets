import pandas as pd
# Lexicon class


class lexicon:
    def __init__(self, emotion, wordlex):
        self.emo = emotion
        self.wordlex = wordlex


def clean_lexicon(word):
    if word[0] == "#" or word[0] == "@":
        word = word[1:]
    return word


def datareader(path_to_file, Elex=True):
    if Elex:
        lexicon = pd.read_csv(path_to_file, sep="\t", index_col=False, names=["emotion", "word", "score"])
        return lexicon
    else:
        lexicon = pd.read_csv(path_to_file, sep="\t", index_col=False, names=["word", "score", "pos", "neg"])
        return lexicon


def load_lexicon(df, Elex=True):
    lexicons = []
    for i in df.index:
        df.at[i, "word"] = clean_lexicon(str(df.at[i, "word"]))

    if Elex:
        for emo in df["emotion"].unique():
            lexd = dict()
            elex = df[df["emotion"] == emo]
            for idx, row in elex.iterrows():
                lexd[row["word"]] = row["score"]
            lexicons.append(lexicon(emo, lexd))
    else:
        lexd = dict()
        for idx, row in df.iterrows():
            lexd[row["word"]] = row["score"]
        lexicons.append(lexicon(emotion="sentiment", wordlex=lexd))
    return lexicons
