##
import numpy as np
from DataParser import raw_to_panda
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import pandas as pd
from Transformations import principle_component_analysis, dmlmj


def main(data_path, embedding, aggregation_method, mlm_method, evaluation=0, transformation = 0):
    """Given a dataset, embedding, aggregationMethod, Machine learning method and an evaluation
       metric the function calculates the pearson correlation coefficient of the model with the
       provided settings.

    Parameters:
        data_path (string): The path to the input dataset
        embedding (Embedding): The embedding method
        aggregation_method (function): Generates the tweet vectors
        mlm_method (MLMmethod): The machine learning method
        evaluation (Int): If = 0 use gold labels, if >0 use as cross-validation amount
        transformation (Int): If = 0 no transformation, if = 1 pca, if 2 dlmjm.

    Returns:
        result (tuple): The averaged Pearson Correlation coefficient with the corresponding p-values

    """
    data_frame = generate_tweet_vector(data_path, embedding, aggregation_method)

    if evaluation > 0:
        result = validate(evaluation, mlm_method, data_frame, transformation)
    else:
        result = 0

    return result


def generate_tweet_vector(data_path, embedding, aggregation_method):
    """Given a dataset, embedding, aggregationMethod, the function generates a DataFrame with TweetVectors

    Parameters:
        data_path (string): The path to the input dataset
        embedding (Embedding): The embedding method
        aggregation_method (function): Generates the tweet vectors

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


def validate(k, method, df, transformation=0):
    """Performs a k-fold crossvalidation on the data for a given method.

    Parameters:
        k (int >0): The number of folds to be performed
        method (MLMethod ) : The machine learning method
        df (pandas DataFrame) : The data on which to perform CV
        transformation (Int): If = 0 no transformation, if = 1 pca, if 2 dlmjm.

    Returns:
        (float, list:floats) = The average Pearson Correlation coefficient and the corresponging p-values

    """
    np.random.seed(3)
    kf = KFold(n_splits=k)
    val_results = []
    p_values = []
    if transformation == 1:
        principle_component_analysis(df)
    for train_index, test_index in kf.split(df):
        if transformation == 2:
            dmlmj(train, test)
        train, test = df.loc[train_index], df.loc[test_index]
        r, p = method.result(test, train)
        val_results.append(r)
        p_values.append(p)
    return sum(val_results) / k, p_values


def plot(data_frame, title):
    df = pd.DataFrame(item for item in data_frame["Vector"])
    df["Label"] = data_frame["Label"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Vector 1', fontsize=15)
    ax.set_ylabel('Vector 2', fontsize=15)
    ax.set_title(title, fontsize=20)

    targets = [0, 1, 2, 3]
    colors = ['r', 'b', 'g', 'y']
    for target, color in zip(targets, colors):
        indices_to_keep = df['Label'] == target
        ax.scatter(df.loc[indices_to_keep, 0]
                   , df.loc[indices_to_keep, 1]
                   , c=color
                   , s=50)
    ax.legend(["0: Not present", "1", "2", "3: Very present"])
    ax.grid()
