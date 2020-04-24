##
import numpy as np
from gensim.models import KeyedVectors
import gluonnlp as nlp
import mxnet as mx
import time


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
        model, vocab = nlp.model.get_model('bert_12_768_12', dataset_name=path,
                                           use_classifier=False, use_decoder=False)
        tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
        transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.transform = transform
        super(Bert, self).__init__(name, path)

    def generate_embedding(self, path):
        model, vocab = nlp.model.get_model('bert_24_1024_16', dataset_name=path, use_classifier=False, use_decoder=False)
        return model

    def generate_word_vectors(self, row, cls=False):
        sample = self.transform([row["TweetText"]])
        words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]])
        text = [self.vocab.bos_token] + self.tokenizer(row["TweetText"]) + [self.vocab.eos_token]
        seq_encoding, cls_encoding = self.embed(words, segments, valid_len)
        # Since Bert embedds some unkown words by cutting them up, here we take the average of the cut up word vectors,
        # hereby obtaining the word vectors for the full words
        if cls:
            return cls_encoding[0]
        else:
            vecs = []
            for i, t in enumerate(text):
                if t is not None:
                    if t[0] == "#":
                        vecs[-1] += seq_encoding[0][i]
                        vecs[-1] /= 2
                    else:
                        vecs.append(seq_encoding[0][i])
        return [v.asnumpy() for v in vecs]


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