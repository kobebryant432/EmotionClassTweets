##
import numpy as np
import scipy.stats
from DataParser import raw_to_panda
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import pandas as pd
from Transformations import principle_component_analysis, dmlmj
from tqdm import tqdm
from AggregationMethods import sentence_avg


def main(data_frame,  mlm_method, evaluation=0, transformation=(0, 2),  gold_data_frame=None):
    """Given a dataset, embedding, aggregationMethod, Machine learning method and an evaluation
       metric the function calculates the pearson correlation coefficient of the model with the
       provided settings.

    Parameters:
        data_path (string): The path to the input dataset
        embedding (EmbeddingMethod): The embedding method
        aggregation_method (function): Generates the tweet vectors
        mlm_method (MLMmethod): The machine learning method
        evaluation (Int): If = 0 use gold labels, if >0 use as cross-validation amount
        gold_data (String): The path to the gold data set in the case of final testing used only if evaluation = 0
        transformation (Int): If = 0 no transformation, if = 1 pca, if 2 dlmjm.

    Returns:
        result (tuple): The averaged Pearson Correlation coefficient with the corresponding p-values

    """

    # If a cross-validation value k is given do cross validation
    if evaluation > 0:
        result = validate(evaluation, mlm_method, data_frame, transformation)
    # If no k but gold_data is given, perform final evaluation on gold labels
    elif gold_data_frame is not None:
        result = gold_test(data_frame, gold_data_frame, mlm_method, transformation)
    # Bad input
    else:
        return "No gold data or no evaluation=k for cross-validation"
    return result


def generate_tweet_vector(data_path, embedding, aggregation_method=sentence_avg, lexicons=None, cls=None):
    """Given a dataset, embedding, aggregationMethod, the function generates a DataFrame with TweetVectors

    Parameters:
        data_path (string): The path to the input dataset
        embedding (EmbeddingMethod): The embedding method
        aggregation_method (function): Generates the tweet vectors

    Returns:
        pandas DataFrame: The panda DataFrame

    """
    data_frame = raw_to_panda(data_path)
    tqdm.pandas()
    if cls:
        data_frame["Vector"] = data_frame.progress_apply(lambda row: embedding.generate_word_vectors(row, lexicons, cls=True), axis=1)
    # Transforming the raw data to a panda data frame
    else:
        # Generating the word vectors, calculating the tweet vector with the given aggregation method
        data_frame["Vector"] = data_frame.progress_apply(lambda row: aggregation_method(embedding.generate_word_vectors(row, lexicons), row), axis=1)

    return data_frame


def validate(k, method, df, transformation=(0, 2)):
    """Performs a k-fold cross-validation on the data for a given method.

    Parameters:
        k (int >0): The number of folds to be performed
        method (MLMethod) : The machine learning method
        df (pandas DataFrame) : The data on which to perform CV
        transformation (Int): If = 0 no transformation, if = 1 pca, if 2 dlmjm.

    Returns:
        (float, list:floats) = The average Pearson Correlation coefficient and the corresponging p-values

    """
    print("Beginning " + str(k) + "-fold cross validation on: " + method.name)
    np.random.seed(3)
    kf = KFold(n_splits=k)
    all_test_label = []
    all_predicted_label = []
    for train_index, test_index in tqdm(kf.split(df)):
        train, test = df.loc[train_index], df.loc[test_index]
        if transformation[0] == 1:
            print("Performing PCA")
            data_frame = pd.concat([train, test], keys=['x', 'y'])
            principle_component_analysis(data_frame, dim=transformation[1])
            train, test = data_frame.loc['x'], data_frame.loc['y']
        if transformation[0] == 2:
            print("Performing dmlmj")
            dmlmj(train, test, dim=transformation[1])
        y_test, y_pred = method.result(test, train)
        for y in y_test: all_test_label.append(y)
        for y in y_pred: all_predicted_label.append(y)
    return scipy.stats.pearsonr(all_predicted_label, all_test_label)


def gold_test(train, test, mlm_method, transformation):
    if transformation[0] == 1:
        data_frame = pd.concat([train, test], keys=['x', 'y'])
        principle_component_analysis(data_frame, dim=transformation[1])
        train, test = data_frame.loc['x'], data_frame.loc['y']
    elif transformation[0] == 2:
        dmlmj(train, test, dim=transformation[1])
    r, p = mlm_method.result(train, test)
    return scipy.stats.pearsonr(r, p)


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
