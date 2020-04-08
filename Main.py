##
import numpy as np
from DataParser import raw_to_panda
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd


def main(data_path, embedding, aggregation_method, mlm_method, evaluation=0):
    """Given a dataset, embedding, aggregationMethod, Machine learning method and an evaluation
       metric the function calculates the pearson correlation coefficient of the model with the
       provided settings.

    Parameters:
        Dataset (string): The path to the input dataset
        Embedding (Embedding): The embedding method
        AggregationMethod (function): Generates the tweet vectors
        MLMmethod (MLMmethod): The machine learning method
        Evaluation metric(Int): If = 0 use gold labels, if >0 use as cross-validation amount

    Returns:
        result (tuple): The averaged Pearson Correlation coefficient with the corresponding p-values

    """
    data_frame = generate_tweet_vector(data_path, embedding, aggregation_method)

    if evaluation > 0:
        result = validate(evaluation, mlm_method, data_frame)
    else:
        result = 0

    return result


def generate_tweet_vector(data_path, embedding, aggregation_method):
    """Given a dataset, embedding, aggregationMethod, the function generates a DataFrame with TweetVectors

    Parameters:
        Dataset (string): The path to the input dataset
        Embedding (Embedding): The embedding method
        AggregationMethod (function): Generates the tweet vectors

    Returns:
        pandas DataFrame: The panda DataFrame

    """
    # Transforming the raw data to a panda data frame
    data_frame = raw_to_panda(data_path)

    # Generating the word vectors, calculating the tweet vector with the given aggregation method
    tweet_vectors = []
    for index, row in data_frame.iterrows():
        tweet_vectors.append(aggregation_method(embedding.generate_word_vectors(row), row))
    data_frame["Vector"] = tweet_vectors

    return data_frame


def principle_component_analysis(data_frame, name):
    """Principle component analysis on the TweetVectors, generating 2-D plots

    Parameters:
        data_frame (pandas DataFrame): The pandas DataFrame containing the columns ["Label"] and ["Vector"]
        name (str) : The name of the plot

    """
    pca = PCA(n_components=2)
    sc = StandardScaler()
    y = data_frame.loc[:, ["Label"]].values
    x = pd.DataFrame(data_frame["Vector"].tolist())

    x = sc.fit_transform(x)
    principlecomponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principlecomponents, columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, data_frame[["Label"]]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA ' + name, fontsize=20)

    targets = [0, 1, 2, 3]
    colors = ['r', 'b', 'g', 'y']
    for target, color in zip(targets, colors):
        indices_to_keep = finalDf['Label'] == target
        ax.scatter(finalDf.loc[indices_to_keep, 'principal component 1']
                   , finalDf.loc[indices_to_keep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(["0: Not present", "1", "2", "3: Very present"])
    ax.grid()

def validate(k, method, df):
    """Performs a k-fold crossvalidation on the data for a given method.

    Parameters:
        k (int >0): The number of folds to be performed
        method (MLMethod ) : The machine learning method
        df (pandas DataFrame) : The data on which to perform CV

    Returns:
        (float, list:floats) = The average Pearson Correlation coefficient and the corresponging p-values

    """
    np.random.seed(3)
    kf = KFold(n_splits=k)
    val_results = []
    p_values = []
    for train_index, test_index in kf.split(df):
        train, test = df.loc[train_index], df.loc[test_index]
        r, p = method.result(test, train)
        val_results.append(r)
        p_values.append(p)
    return sum(val_results) / k, p_values


