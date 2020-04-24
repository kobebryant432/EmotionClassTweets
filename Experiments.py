##
#Imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Main import *
from Transformations import principle_component_analysis, dmlmj
import Embedding as em
import Lexicon as l
from AggregationMethods import n_avg, lexicon_avg, all_lexicon_avg,hashtag
import MLMethod as M
from collections import defaultdict
import pandas as pd


##
# Insert the paths to the raw datasets, test and training+development
emotions = ["anger", "fear", "joy", "sadness"]
training_data_paths = []
for emo in emotions:
    d = r"Thesis - Inputdata\2018-EI-oc-En-" + emo + "-dev-and-train.txt"
    training_data_paths.append(d)

test_data_paths = []
for emo in emotions:
    d = r"Thesis - Inputdata\2018-EI-oc-En-" + emo + "-test-gold.txt"
    test_data_paths.append(d)
##
# Insert paths/type of embeddings(bert, Glove, Word2vec: Skipgram)
path_bert = 'book_corpus_wiki_en_cased'
path_glove = r"Thesis - Embeddings\GloVe\glove.6B.300d.w2vformat.txt"
path_word2vec = r"Thesis - Embeddings\Word2Vec\GoogleNews-vectors-negative300.bin"

# loading the embeding methods
glove = em.WordToVec("glove", path_glove)
word_two_vec = em.WordToVec("word2vec", path_word2vec)
bert = em.Bert("bert", path_bert)
embeds = [glove, word_two_vec, bert]
##
# Insert path to the Lexicon Files:
print("Loading Lexicons")
path_emo_lex = r"Lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2\NRC-Hashtag-Emotion-Lexicon-v0.2.txt"
path_sen_lex = r"Lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0\HS-unigrams.txt"

# Generating the lexicons
lexi = l.load_lexicon(l.datareader(path_emo_lex))
lexi += l.load_lexicon(l.datareader(path_sen_lex, Elex=False), Elex=False)
print("Complete")
##
# Loading aggregation methods
n_avg = n_avg
all_lexi = lambda x, y: all_lexicon_avg(x, y, lexi[:-1])
esla_sentiment = lambda x, y: lexicon_avg(x, y, lexi[-1])
esla_fear = lambda x, y: lexicon_avg(x, y, lexi[1])
agg_methods = {"NormalAvg": n_avg, "HashAvg": hashtag, "All_lexi": all_lexi, "Emotion_specific": esla_fear, "Sentiment": esla_sentiment}

##
from frlearn.neighbours.owa_operators import strict, additive, invadd
# For the baseline methods determine optimal k, using the optimal weighting schemes for FRNN per dataset.
methods = [
[lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA:invadd", invadd(x), invadd(x))],
[lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA:add+strict", additive(x), strict)],
[lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA:invadd", invadd(x), invadd(x))],
[lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA:add+strict", additive(x), strict)],
 ]
current_methods = {emo: m for emo, m in zip(emotions, methods)}

names = ["KNN", "FRNN"]
results = [defaultdict(list),defaultdict(list)]
for emo, path in zip(emotions, training_data_paths):
    dat = generate_tweet_vector(path, glove, n_avg)
    for k in [1, 5, 10, 15, 20]:
        print("Testing using " + str(k) + " nearest neighbours")
        for i in range(2):
            results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i](k), evaluation=20)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    resultsdf["Totals"] = resultsdf.sum(axis=1)/4
    resultsdf["Nearest Neighbours"] = [1, 5, 10, 15, 20]
    resultsdf.to_excel(r"Final Output\results_nearestneighbours_" + names[i] + ".xlsx")
##
k=20
# With the above found k, look for the optimal values embedding
methods = [
[M.NearestNeighbour(k), M.FRNN_OWA("OWA:invadd", invadd(k), invadd(k))],
[M.NearestNeighbour(k), M.FRNN_OWA("OWA:add+strict", additive(k), strict)],
[M.NearestNeighbour(k), M.FRNN_OWA("OWA:invadd", invadd(k), invadd(k))],
[M.NearestNeighbour(k), M.FRNN_OWA("OWA:add+strict", additive(k), strict)],
 ]
current_methods = {emo: m for emo, m in zip(emotions, methods)}

results = [defaultdict(list),defaultdict(list)]
for embed in embeds:
    print("Evaluating embedding method: " + embed.name)
    for emo, path in zip(emotions, training_data_paths):
        dat = generate_tweet_vector(path, embed, n_avg)
        for i in range(2):
            results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=20)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    resultsdf["Totals"] = resultsdf.sum(axis=1)/4
    resultsdf["Embedmethods"] = [em.name for em in embeds]
    resultsdf.to_excel(r"Final Output\results_embeddings_" + names[i] + ".xlsx")

##
# Optimal embedding:
from statistics import mean, median
embedding = glove
#The lexicons defining the different aggregation methods
agg_methods = {"NormalAvg": n_avg,"Hashtag":hashtag, "All_lexi": lexi[:-1], "Emotion_specific": dict(zip(emotions, [lexi[2], lexi[1], lexi[6], lexi[5]])),  "Sentiment": lexi[-1]}

c_functions = [lambda lexi: 1,
               lambda lexi: min(lexi.wordlex.values(), key=lambda x:abs(x)),
               lambda lexi: min(lexi.wordlex.values(), key=lambda x:abs(x)),
               lambda lexi: mean([abs(x) for x in lexi.wordlex.values()]),
               lambda lexi: median([abs(x) for x in lexi.wordlex.values()])]

results = [defaultdict(list), defaultdict(list)]
for name, agg in agg_methods.items():
    print("Evaluating aggregation method: " + name)
    if agg == n_avg or agg == hashtag:
        for emo, path in zip(emotions, training_data_paths):
            dat = generate_tweet_vector(path, embedding, agg)
            for i in range(2):
                results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=20)[0])
    else:
        for c in c_functions:
            for emo, path in zip(emotions, training_data_paths):
                lexicon_used = agg
                if name == "Emotion_specific":
                    lexicon_used = agg[emo]
                if name != "All_lexi":
                    dat = generate_tweet_vector(path, embedding, aggregation_method=lambda x, y: lexicon_avg(x, y, lexicon_used, c(lexicon_used)))
                else:
                    c_value = mean(c(l) for l in lexicon_used)
                    dat = generate_tweet_vector(path, embedding, aggregation_method=lambda x, y: all_lexicon_avg(x,y, lexicon_used, c_value))
                for i in range(2):
                    results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=20)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    resultsdf["Totals"] = resultsdf.sum(axis=1)/4
    resultsdf.to_excel(r"Final Output\results_aggmethods_" + names[i] + ".xlsx")


## Finding out if a transformation is helpful (to x dimensions)
# Optimal aggmethod:
aggregation_method = n_avg

results = [defaultdict(list),defaultdict(list)]
transformations = ["None", "PCA", "DMLMJ"]
for t, name in enumerate(transformations):
    print("Evaluating transformation method method: " + name)
    for emo, path in zip(emotions, training_data_paths):
        dat = generate_tweet_vector(path, embedding, aggregation_method)
        for i in range(2):
            results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=20, transformation=t)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    resultsdf["Totals"] = resultsdf.sum(axis=1)/4
    resultsdf["Transformation methods"] = transformations
    resultsdf.to_excel(r"Final Output\results_transformations_" + names[i] + ".xlsx")
##
# from matplotlib import pyplot as plt
# df = pd.DataFrame()
# for name, agg in agg_methods.items():
#     dat = generate_tweet_vector("Thesis - Inputdata/aggtestdata", glove, agg)
#     train = generate_tweet_vector(training_data_paths[1], glove, agg)
#     dmlmj(dat)
#     df[name] = dat["Vector"]
# df["Label"] = dat["Label"]
# df["TweetText"] = dat["TweetText"]
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Vector 1', fontsize=15)
# ax.set_ylabel('Vector 2', fontsize=15)
# ax.set_title("c=0", fontsize=20)
#
#
# colors = ['r', 'b', 'g', 'y']
# markers = ["s", "o", "P", "*"]
# for i, mark in zip(range(len(df["Label"])), markers):
#     for name, agg, color in zip(agg_methods.keys(), agg_methods.values(), colors):
#         ax.scatter(df[name][i][0]
#                        , df[name][i][1]
#                        , c=color
#                        , marker=mark
#                        , s=50)
#
# f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
# handles = [f(".", color) for color in colors]
# handles += [f(marker, "k") for marker in markers]
#
# labels = list(agg_methods.keys()) + df["TweetText"].tolist()
#
# ax.legend(handles, labels, loc='best')
# ax.grid()
