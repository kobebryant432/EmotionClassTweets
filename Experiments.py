##
#Imports
from frlearn.utils.owa_operators import strict, additive, invadd, exponential, mean, trimmed
from AggregationMethods import n_avg, lexicon_avg, all_lexicon_avg, hashtag
from Main import *
from collections import defaultdict
import Embedding as em
import Lexicon as l
import MLMethod as M
import pandas as pd
import scipy.stats as st
import scikit_posthocs as hocs
import warnings
from statistics import median
from statistics import mean as me
warnings.simplefilter(action='ignore', category=FutureWarning)

##
# Insert the paths to the raw datasets, test and training+development
emotions = ["Anger", "Fear", "Joy", "Sadness"]
training_data_paths = []
for emo in emotions:
    d = r"Thesis - Inputdata\2018-EI-oc-En-" + emo + "-dev-and-train.txt"
    training_data_paths.append(d)

test_data_paths = []
for emo in emotions:
    d = r"Thesis - Inputdata\2018-EI-oc-En-" + emo + "-test-gold.txt"
    test_data_paths.append(d)
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
# Loading aggregation methods
n_avg = n_avg
all_lexi = lambda x, y: all_lexicon_avg(x, y, lexi[:-1])
esla_sentiment = lambda x, y: lexicon_avg(x, y, lexi[-1])
esla_fear = lambda x, y: lexicon_avg(x, y, lexi[1])
agg_methods = {"NormalAvg": n_avg, "HashAvg": hashtag, "All_lexi": all_lexi, "Emotion_specific": esla_fear, "Sentiment": esla_sentiment}

##
# Optimizing for the best weighting strategy
methods = [
    lambda x: M.FRNN_OWA("OWA", x, x, 15, metric="euclidean"),
    lambda x: M.FRNN_OWA("OWA", x, x, 15, metric="euclidean"),
    lambda x: M.FRNN_OWA("OWA", x, x, 15, metric="euclidean"),
    lambda x: M.FRNN_OWA("OWA", x, x, 15, metric="euclidean"),
 ]
Recomendations = [
    M.FRNN_OWA("OWA", invadd(), invadd(), 15, metric="euclidean"),
    M.FRNN_OWA("OWA", additive(), strict(), 15, metric="euclidean"),
    M.FRNN_OWA("OWA", invadd(), invadd(), 15, metric="euclidean"),
    M.FRNN_OWA("OWA", additive(), strict(), 15, metric="euclidean"),
 ]
current_methods = {emo: m for emo, m in zip(emotions, methods)}
recomended_methods = {emo: m for emo, m in zip(emotions, Recomendations)}

results = defaultdict(list)
for emo, path in zip(emotions, training_data_paths):
    dat = generate_tweet_vector(path, glove, n_avg)
    for owa in [strict(), invadd(), additive(), exponential(), mean(), trimmed()]:
        print("Testing using different OWA weights")
        results[emo].append(main(data_frame=dat, mlm_method=current_methods[emo](owa), evaluation=5)[0])
    results[emo].append(main(data_frame=dat, mlm_method=recomended_methods[emo], evaluation=5)[0])

resultsdf = pd.DataFrame(results)
friedman = st.friedmanchisquare(*resultsdf.values)
print("The statistic for FRNN")
print(friedman)
if friedman.pvalue < 0.05:
    p_values = hocs.posthoc_conover_friedman(resultsdf.T,  p_adjust="holm")
    p_values.to_excel(r"Final Output\results_OWA_operators_FRNN_p_values.xlsx")
ranks = pd.DataFrame(columns=resultsdf.keys())
for key in resultsdf.keys():
    ranks[key] = resultsdf[key].rank(ascending=False)
resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
resultsdf["Ranks"] = ranks.mean(axis=1)
resultsdf["Nearest Neighbours"] = ["Strict", "Invadd", "Additive", "Exponential", "Mean", "Trimmed", "Recommended"]
resultsdf.to_excel(r"Final Output\results_OWA_operators_FRNN.xlsx")
##

# For the baseline methods determine optimal k, using the optimal weighting schemes for FRNN per dataset.
methods = [
    [lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA", mean(), mean(), x, metric="euclidean")],
    [lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA", invadd(), invadd(), x, metric="euclidean")],
    [lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA", mean(), mean(), x, metric="euclidean")],
    [lambda x: M.NearestNeighbour(x), lambda x: M.FRNN_OWA("OWA", mean(), mean(), x, metric="euclidean")],
 ]
current_methods = {emo: m for emo, m in zip(emotions, methods)}

names = ["KNN", "FRNN"]
results = [defaultdict(list), defaultdict(list)]
for emo, path in zip(emotions, training_data_paths):
    dat = generate_tweet_vector(path, glove, n_avg)
    for k in [1, 2, 3, 5, 10, 15, 20]:
        print("Testing using " + str(k) + " nearest neighbours")
        for i in range(2):
            results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i](k), evaluation=5)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    friedman = st.friedmanchisquare(*resultsdf.values)
    print("The statistic for " + names[i])
    print(friedman)
    if friedman.pvalue < 0.05:
        p_values = hocs.posthoc_conover_friedman(resultsdf.T, p_adjust="holm")
        p_values.to_excel(r"Final Output\results_nearestneighbours_" + names[i] + "_p_values.xlsx")
    ranks = pd.DataFrame(columns=resultsdf.keys())
    for key in resultsdf.keys():
        ranks[key] = resultsdf[key].rank(ascending=False)
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Ranks"] = ranks.mean(axis=1)
    resultsdf["Nearest Neighbours"] = [1, 2, 3, 5, 10, 15, 20]
    resultsdf.to_excel(r"Final Output\results_nearestneighbours_" + names[i] + ".xlsx")

##
# Checking which embedding method works best
# With the above found k, look for the optimal value for the embedding
methods = [
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
    [M.NearestNeighbour(1), M.FRNN_OWA("OWA", invadd(), invadd(), 20, metric="euclidean")],
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
 ]
current_methods = {emo: m for emo, m in zip(emotions, methods)}

results = [defaultdict(list), defaultdict(list)]
for embed in embeds:
    print("Evaluating embedding method: " + embed.name)
    for emo, path in zip(emotions, training_data_paths):
        if embed.name == "bert":
            dat_bert_cls = generate_tweet_vector(path, embed, n_avg, cls=True)
            for i in range(2):
                results[i][emo].append(main(data_frame=dat_bert_cls, mlm_method=current_methods[emo][i], evaluation=5)[0])
        dat = generate_tweet_vector(path, embed, n_avg)
        for i in range(2):
            results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5)[0])
        dat_lex = generate_tweet_vector(path, embed, n_avg, lexi)
        for i in range(2):
            results[i][emo].append(main(data_frame=dat_lex, mlm_method=current_methods[emo][i], evaluation=5)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    friedman = st.friedmanchisquare(*resultsdf.values)
    print("The statistic for " + names[i])
    print(friedman)
    if friedman.pvalue < 0.05:
        p_values = hocs.posthoc_conover_friedman(resultsdf.T, p_adjust="holm")
        p_values.to_excel(r"Final Output\results_embeddings_" + names[i] + "_p_values.xlsx")
    ranks = pd.DataFrame(columns=resultsdf.keys())
    for key in resultsdf.keys():
        ranks[key] = resultsdf[key].rank(ascending=False)
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Ranks"] = ranks.mean(axis=1)
    resultsdf["Embedmethods"] = ["GloVe", "GloVe_Lexicon", "Word2vec", "Word2vec_lexicon", "Bert_Sentence", "Bert", "Bert_lexicon", ]
    resultsdf.to_excel(r"Final Output\results_embeddings_" + names[i] + ".xlsx")
##
# Checking which aggregation method is the best
# Optimal embedding from previous results:
embedding = bert
# The lexicons defining the different aggregation methods
agg_methods = {"NormalAvg": n_avg, "Hashtag": hashtag, "All_lexi": lexi[:-1], "Emotion_specific": dict(zip(emotions, [lexi[2], lexi[1], lexi[6], lexi[5]])),  "Sentiment": lexi[-1]}

# The values for c that we will be testing
c_functions = [lambda lexi: 1,
               lambda lexi: min(lexi.wordlex.values(), key=lambda x:abs(x)),
               lambda lexi: np.percentile([abs(x) for x in lexi.wordlex.values()], 20, interpolation='midpoint'),
               lambda lexi: median([abs(x) for x in lexi.wordlex.values()])]

results = [defaultdict(list), defaultdict(list)]
for name, agg in agg_methods.items():
    print("Evaluating aggregation method: " + name)
    if agg == n_avg:
        for emo, path in zip(emotions, training_data_paths):
            dat = generate_tweet_vector(path, embedding, agg, lexi)
            for i in range(2):
                results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5)[0])
    elif agg == hashtag:
        for emo, path in zip(emotions, training_data_paths):
            for c in [2, 4, 8, 16]:
                dat = generate_tweet_vector(path, embedding, lambda x, y: hashtag(x, y, c=c), lexi)
                for i in range(2):
                    results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5)[0])
    else:
        for c in c_functions:
            for emo, path in zip(emotions, training_data_paths):
                lexicon_used = agg
                if name == "Emotion_specific":
                    lexicon_used = agg[emo]
                if name != "All_lexi":
                    dat = generate_tweet_vector(path, embedding, aggregation_method=lambda x, y: lexicon_avg(x, y, lexicon_used, c(lexicon_used)), lexicons=lexi)
                else:
                    c_value = me(c(l) for l in lexicon_used)
                    dat = generate_tweet_vector(path, embedding, aggregation_method=lambda x, y: all_lexicon_avg(x, y, lexicon_used, c_value), lexicons=lexi)
                for i in range(2):
                    results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    friedman = st.friedmanchisquare(*resultsdf.values)
    print("The statistic for " + names[i])
    print(friedman)
    if friedman.pvalue < 0.05:
        p_values = hocs.posthoc_conover_friedman(resultsdf.T, p_adjust="holm")
        p_values.to_excel(r"Final Output\results_aggmethods_" + names[i] + "_p_values.xlsx")
    ranks = pd.DataFrame(columns=resultsdf.keys())
    for key in resultsdf.keys():
        ranks[key] = resultsdf[key].rank(ascending=False)
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Ranks"] = ranks.mean(axis=1)
    resultsdf.to_excel(r"Final Output\results_aggmethods_" + names[i] + ".xlsx")
##
# Finding out if a transformation is helpful (to x dimensions)
# Optimal aggmethod:
methods = [
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
    [M.NearestNeighbour(1), M.FRNN_OWA("OWA", invadd(), invadd(), 20, metric="euclidean")],
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
 ]
current_methods = {emo: m for emo, m in zip(emotions, methods)}
embedding = bert
aggregation_method = dict(zip(emotions, [lambda x, y: lexicon_avg(x, y, l, np.percentile([abs(x) for x in l.wordlex.values()], 20, interpolation='midpoint')) for l in [lexi[2], lexi[1], lexi[6], lexi[5]]]))

results = [defaultdict(list), defaultdict(list)]
transformations = ["None", "PCA", "DMLMJ"]
names = ["KNN", "FRNN"]

for emo, path in zip(emotions, training_data_paths):
    print("Evaluating " + emo + " dataset")
    for i in range(2):
        agg = aggregation_method[emo]
        for t, name in reversed(list(enumerate(transformations))):
            print("Transformation " + name)
            if t != 0:
                for dim in [2, 8, 32, 128, 300]:
                    dat = generate_tweet_vector(path, embedding, aggregation_method=agg, lexicons=lexi)
                    print("Dim " + str(dim))
                    results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5, transformation=(t, dim))[0])
            else:
                dat = generate_tweet_vector(path, embedding, aggregation_method=agg, lexicons=lexi)
                results[i][emo].append(main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    friedman = st.friedmanchisquare(*resultsdf.values)
    print("The statistic for " + names[i])
    print(friedman)
    if friedman.pvalue < 0.05:
        p_values = hocs.posthoc_conover_friedman(resultsdf.T, p_adjust="holm")
        p_values.to_excel(r"Final Output\results_transformations_" + names[i] + "_p_values.xlsx")
    ranks = pd.DataFrame(columns=resultsdf.keys())
    for key in resultsdf.keys():
        ranks[key] = resultsdf[key].rank(ascending=False)
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Ranks"] = ranks.mean(axis=1)
    resultsdf["Transformation methods"] = ["None"] + [str(300), str(128), str(32), str(8), str(2)]*2
    resultsdf.to_excel(r"Final Output\results_transformations_" + names[i] + ".xlsx")

##
for emo, path in zip(emotions, training_data_paths):
    print("Evaluating " + emo + " dataset")
    for i in range(2):
        agg = aggregation_method[emo]
        dat = generate_tweet_vector(path, embedding, aggregation_method=agg, lexicons=lexi)
        results[i][emo].append(
            main(data_frame=dat, mlm_method=current_methods[emo][i], evaluation=5, transformation=(2, 2))[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    friedman = st.friedmanchisquare(*resultsdf.values)
    print("The statistic for " + names[i])
    print(friedman)
    if friedman.pvalue < 0.05:
        p_values = hocs.posthoc_conover_friedman(resultsdf.T, p_adjust="holm")
        p_values.to_excel(r"Final Output\results_dmlmjs_" + names[i] + "_p_values.xlsx")
    ranks = pd.DataFrame(columns=resultsdf.keys())
    for key in resultsdf.keys():
        ranks[key] = resultsdf[key].rank(ascending=False)
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Ranks"] = ranks.mean(axis=1)
    resultsdf["Transformation methods"] = ["None"] + [str(300), str(128), str(32), str(8), str(2)]*2 + ["CrossValidated"]
    resultsdf.to_excel(r"Final Output\results_dmlmjs_" + names[i] + ".xlsx")
##
# The gold test for the final model
names = ["KNN", "FRNN"]
final_embed = [word_two_vec, bert]
final_aggregation = [lambda x, y: lexicon_avg(x, y, lexi[-1], 1),
                      dict(zip(emotions, [lambda x, y: lexicon_avg(x, y, l, np.percentile([abs(x) for x in l.wordlex.values()], 20, interpolation='midpoint')) for l in [lexi[2], lexi[1], lexi[6], lexi[5]]]))]
final_transformation = [(0, 1), (0, 1)]
final_methods = [
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
    [M.NearestNeighbour(1), M.FRNN_OWA("OWA", invadd(), invadd(), 20, metric="euclidean")],
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
    [M.NearestNeighbour(20), M.FRNN_OWA("OWA", mean(), mean(), 20, metric="euclidean")],
 ]
current_methods = {emo: m for emo, m in zip(emotions, final_methods)}
results = [defaultdict(list), defaultdict(list)]

for emo, train_path, test_path in zip(emotions, training_data_paths, test_data_paths):
    for i in range(2):
        agg = final_aggregation[i]
        if i ==1: agg = agg[emo]
        train_data = generate_tweet_vector(train_path, final_embed[i], agg, lexicons=lexi)
        test_data = generate_tweet_vector(test_path, final_embed[i], agg, lexicons=lexi)

        results[i][emo].append(main(data_frame=train_data, mlm_method=current_methods[emo][i], evaluation=0, transformation=final_transformation[i], gold_data_frame=test_data)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Final Results"] = ["Scores:"]
    resultsdf.to_excel(r"Final Output\final_results_" + names[i] + ".xlsx")
##
# The gold test for the baseline model
names = ["KNN", "FRNN"]
final_embed = [glove, glove]
final_aggregation = [n_avg, n_avg]
final_transformation = [(0, 1), (0, 1)]
final_methods = [
    [M.NearestNeighbour(1), M.FRNN_OWA("OWA", additive(), additive(), 10)],
    [M.NearestNeighbour(1), M.FRNN_OWA("OWA", additive(), additive(), 15)],
    [M.NearestNeighbour(15), M.FRNN_OWA("OWA", additive(), additive(), 20)],
    [M.NearestNeighbour(15), M.FRNN_OWA("OWA", additive(), additive(), 20)],
 ]
current_methods = {emo: m for emo, m in zip(emotions, final_methods)}
results = [defaultdict(list), defaultdict(list)]

for emo, train_path, test_path in zip(emotions, training_data_paths, test_data_paths):
    for i in range(2):
        agg = final_aggregation[i]
        train_data = generate_tweet_vector(train_path, final_embed[i], agg, lexicons=lexi)
        test_data = generate_tweet_vector(test_path, final_embed[i], agg, lexicons=lexi)

        results[i][emo].append(main(data_frame=train_data, mlm_method=current_methods[emo][i], evaluation=0, transformation=final_transformation[i], gold_data_frame=test_data)[0])

for i in range(2):
    resultsdf = pd.DataFrame(results[i])
    resultsdf["Totals"] = resultsdf.sum(axis=1) / 4
    resultsdf["Final Results"] = ["Scores:"]
    resultsdf.to_excel(r"Final Output\final_results_baseline" + names[i] + ".xlsx")


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