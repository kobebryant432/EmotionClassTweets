##
from Main import main, generate_tweet_vector, plot
from Transformations import principle_component_analysis, dmlmj
import Embedding as em
import Lexicon as l
from AggregationMethods import n_avg, lexicon_avg, all_lexicon_avg
import MLMethod as M
import FuzzyRough as fr
from collections import defaultdict
import pandas as pd

##
# Insert the paths to the raw datasets
emotions = ["anger", "fear", "joy", "sadness"]
data_paths = []
for emo in emotions:
    d = path_to_file = r"D:\School\Thesis\Thesis - Inputdata\2018-EI-oc-En-" + emo + "-dev-and-train.txt"
    data_paths.append(d)
##
# Insert paths/type of embeddings
path_bert = 'book_corpus_wiki_en_cased'
path_glove = "D:\School\Thesis\Thesis - Embeddings\GloVe\glove.6B.300d.w2vformat.txt"
path_word2vec = "D:\School\Thesis\Thesis - Embeddings\Word2Vec\GoogleNews-vectors-negative300.bin"

# loading the embeding methods
glove = em.WordToVec("glove", path_glove)
# word_two_vec = em.WordToVec("word2vec", path_word2vec)
# bert = em.Bert("bert", path_bert)
# embeds = [glove, word_two_vec, bert]
##
# Lexicons
# Insert path of Lexicon Files:
print("Loading Lexicons")
path_emo_lex = r"D:\School\Thesis\Lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2\NRC-Hashtag-Emotion-Lexicon-v0.2.txt"
path_sen_lex = r"D:\School\Thesis\Lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0\HS-unigrams.txt"

# Generating the lexicons
lexi = l.load_lexicon(l.datareader(path_emo_lex))
lexi += l.load_lexicon(l.datareader(path_sen_lex, Elex=False), Elex=False)
print("Complete")
##
# Loading aggregation methods
n_avg = n_avg
all_lexi = lambda x, y: all_lexicon_avg(x, y, lexi[:-1])
esla_sentiment = lambda x, y: lexicon_avg(x, y, lexi[-1])
esla_anger = lambda x, y: lexicon_avg(x, y, lexi[2])
agg_methods = {"NormalAvg": n_avg, "All_lexi": all_lexi, "Emotion_specific": esla_anger, "Sentiment": esla_sentiment}

##
# Initializing machine learning methods
MLM = [
    M.SVM(),
    M.RandomForest(),
    M.NearestNeighbour(20),
    M.FRNN_OWA("OWA test:", fr.relation1, fr.I_lukasiewicz, fr.T_lukasiewicz, fr.W_add_L, fr.W_add_U, 20)
]
##
# Generating the plots and results
results = defaultdict(list)
#
for name, agg in agg_methods.items():
    dat = generate_tweet_vector(data_paths[0], glove, agg)
    dmlmj(dat)
    plot(dat, "dmlmj")
    results["Aggmethod"].append(name)
    results[MLM[-1].name].append(main(data_paths[0], glove, agg, MLM[-1], 5, 1)[0])

results["Aggmethod"] = results["Aggmethod"][:4]
resultsdf = pd.DataFrame(results)
resultsdf.to_excel(r"C:\Users\Kobe\Desktop\0.000000000001\results.xlsx")
