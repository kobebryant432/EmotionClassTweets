##
from Main import main, generate_tweet_vector, principle_component_analysis

#Insert the paths to the raw datasets
emotions = ["anger", "fear", "joy", "sadness"]
data_paths = []
for emo in emotions:
    d = path_to_file = r"D:\School\Thesis\Thesis - Inputdata\2018-EI-oc-En-"+emo+"-dev-and-train.txt"
    data_paths.append(d)
##
import Embedding as em

#Insert paths/type of embeddings
path_bert = 'book_corpus_wiki_en_cased'
path_glove = "D:\School\Thesis\Thesis - Embeddings\GloVe\glove.6B.300d.w2vformat.txt"
path_word2vec =  "D:\School\Thesis\Thesis - Embeddings\Word2Vec\GoogleNews-vectors-negative300.bin"

#loading the embeding methods
glove = em.WordToVec("glove", path_glove)
#word_two_vec = em.WordToVec("word2vec", path_word2vec)
#bert = em.Bert("bert", path_bert)

#embeds = [glove, word_two_vec, bert]
##
import Lexicon as l
from AggregationMethods import n_avg,lexicon_avg, all_lexicon_avg
#Lexicons
#Insert path of Lexicon Files:
print("Loading Lexicons")
path_emo_lex = r"D:\School\Thesis\Lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2\NRC-Hashtag-Emotion-Lexicon-v0.2.txt"
path_sen_lex = r"D:\School\Thesis\Lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0\HS-unigrams.txt"

#Generating the lexicons
lexi = l.load_lexicon(l.datareader(path_emo_lex))
lexi += l.load_lexicon(l.datareader(path_sen_lex, Elex=False), Elex=False)
print("Complete")
##
#Loading aggregation methods
n_avg = n_avg
all_lexi = lambda x, y: all_lexicon_avg(x, y, lexi[:-1])
esla_sentiment = lambda x, y: lexicon_avg(x, y, lexi[-1])
esla_anger = lambda x, y: lexicon_avg(x, y, lexi[2])
agg_methods = {"NormalAvg": n_avg, "All_lexi": all_lexi, "Emotion_specific": esla_anger, "Sentiment": esla_sentiment}

##
import MLMethod as M
import FRNN
import FuzzyRough as fr
#Initializing Fuzzy-Rough Nearest Neighbour methods
VQRS = lambda x, y, k, r: FRNN.FRNN_VQRS(x, y, k, lambda a: fr.q_linear(a, 0.5, 1), lambda a: fr.q_linear(a, 0, 0.5), r)
OWA = lambda x, y, k, r: FRNN.FRNN_OWA(x, y, k, fr.I_lukasiewicz, fr.T_lukasiewicz, fr.W_exp_L, fr.W_exp_U, r)
FRS = lambda x, y, k, r: FRNN.FRNN_FRS(x, y, k, fr.I_lukasiewicz, fr.T_lukasiewicz, r)

MLM = [
    M.SVM(),
    M.RandomForest(),
    M.NearestNeighbour(20),
    M.FRNN_OWA("OWA test:", fr.relation1, fr.I_lukasiewicz, fr.T_lukasiewicz, fr.W_add_L, fr.W_add_U, 20)
]

##
from collections import defaultdict
import pandas as pd


results = defaultdict(list)
for m in MLM:
    for name, agg in agg_methods.items():
        dat = generate_tweet_vector(data_paths[0], glove, agg)
        principle_component_analysis(dat, name)
        results["Aggmethod"].append(name)
        results[m.name].append(main(data_paths[0], glove, agg, m, 5)[0])

results["Aggmethod"] = results["Aggmethod"][:4]
resultsdf = pd.DataFrame(results)
resultsdf.to_excel(r"C:\Users\Kobe\Desktop\0.000000000001\results.xlsx")