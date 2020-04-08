##
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
from bert_embedding import BertEmbedding


class EmbeddingMethod:
    """
       A class used to represent an embedding method

       ...

       Attributes
       ----------
       name : str
           the name of the embedding method
       embed : <embed>
           the embedding loaded from the path

       Methods
       -------
        generate_embedding(path):
           loads the embedding function/dictionary from the path

        word_vec_generator(row):
            allocates the word vector to a given word

       """

    def __init__(self, name, path):
        self.name = name
        print("started loading model: " + name)
        self.embed = self.generate_embedding(path)

    def generate_embedding(self, path):
        raise NotImplementedError

    def generate_word_vectors(self, row):
        """Given a row in of a data frame, this function generates the word vectors of the the words in that row.

        Parameters:
            row (pandas Series): A single row of the dataframe with ["TweetText"] and ["Words"] columns

        Returns:
            vecs (list): A list of the resulting word vectors.

        """
        raise NotImplementedError


class Bert(EmbeddingMethod):
    def __init__(self, name, path):
        super(Bert, self).__init__(name, path)

    def generate_embedding(self, path):
        return BertEmbedding(model='bert_24_1024_16', dataset_name=path)

    def generate_word_vectors(self, row):
        sentence = row["TweetText"]
        results = self.embed([sentence])
        vecs = []
        for wordvec in results[0][1]:
            vecs.append(wordvec)
        return vecs


class WordToVec(EmbeddingMethod):
    def __init__(self, name, path):
        super(WordToVec, self).__init__(name, path)

    def generate_embedding(self, path):
        b = True
        if self.name == "glove": b = False
        model = KeyedVectors.load_word2vec_format(
            path,
            binary=b
        )
        return model

    def generate_word_vectors(self, row):
        model = self.embed
        vecs = []
        for w in row["Words"]:
            if w in model:
                vecs.append(np.array(model[w]))
            else:
                vecs.append(np.zeros(self.embed.vector_size))
        return vecs


