import pandas as pd
# Lexicon class


class Lexicon:
    """
        A class used to represent a Lexicon

        ...

        Attributes
        ----------
        name : str
            the emotion that this lexicon evaluates/scores
        embed : dict()
            A dict containing the words with respective lexicon scores

        """
    def __init__(self, emotion, wordlex):
        self.emo = emotion
        self.wordlex = wordlex


def clean_lexicon(word):
    """Cleaning the lexicons

        Parameters:
            word (str):  The word to be cleaned

        Returns:
            word (str): The cleaned word
    """
    if word[0] == "#" or word[0] == "@":
        word = word[1:]
    return word


def datareader(path_to_file, Elex=True):
    """Transform the raw input of a lexicon to a pandas DataFrame

        Parameters:
            path_to_file (str):  The path to the lexicon file
            Elex (binary): True if it is an emotion lexicon, False if it is an sentiment lexicon
        Returns:
            lexicon (pandas DataFrame): The resulting DataFrame
    """
    if Elex:
        lexicon = pd.read_csv(path_to_file, sep="\t", index_col=False, names=["emotion", "word", "score"])
        return lexicon
    else:
        lexicon = pd.read_csv(path_to_file, sep="\t", index_col=False, names=["word", "score", "pos", "neg"])
        return lexicon


def load_lexicon(df, Elex=True):
    """Transforms the  pandas DataFrame to the a list of Lexicons
            Parameters:
                df (pandas DataFrame):  The df from datareader
                Elex (binary): True if it is an emotion lexicon, False if it is an sentiment lexicon
            Returns:
                lexicons (list: Lexicons): A list of the lexicons created
        """
    lexicons = []
    for i in df.index:
        df.at[i, "word"] = clean_lexicon(str(df.at[i, "word"]))

    if Elex:
        for emo in df["emotion"].unique():
            lexd = dict()
            elex = df[df["emotion"] == emo]
            for idx, row in elex.iterrows():
                lexd[row["word"]] = row["score"]
            lexicons.append(Lexicon(emo, lexd))
    else:
        lexd = dict()
        for idx, row in df.iterrows():
            lexd[row["word"]] = row["score"]
        lexicons.append(Lexicon(emotion="sentiment", wordlex=lexd))
    return lexicons
