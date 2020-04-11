import os.path
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matlab.engine




def principle_component_analysis(data_frame):
    """Principle component analysis on the TweetVectors, generating 2-D plots

    Parameters:
        test (pandas DataFrame): The pandas DataFrame containing the columns ["Label"] and ["Vector"]
        train (str) (pandas DataFrame): The pandas DataFrame containing the columns ["Label"] and ["Vector"]

    """
    pca = PCA(n_components=2)
    sc = StandardScaler()
    y = data_frame.loc[:, ["Label"]].values
    x = pd.DataFrame(data_frame["Vector"].tolist())
    x = sc.fit_transform(x)
    principlecomponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principlecomponents, columns=['principal component 1', 'principal component 2'])
    data_frame["Vector"] = principalDf.values.tolist()


def dmlmj(train, test=None):
    train1 = pd.DataFrame(item for item in train["Vector"])
    train1["Label"] = train["Label"].values
    train1.to_csv(r"Out\train.csv")

    if test is not None:
        test1 = pd.DataFrame(item for item in test["Vector"])
        test1["Label"] = test["Label"].values
        test1.to_csv(r"Out\test.csv")

    eng = matlab.engine.start_matlab()
    eng.addpath('matlab')
    eng.thesis_distance_learning(nargout=0)

    train_vectors = wait_for_file(r"In\train_in.csv")
    if test is not None:
        test_vectors = wait_for_file(r"In\test_in.csv")

    os.remove(r"Out\train.csv")
    if test is not None:
        os.remove(r"Out\test.csv")

    if test is not None:
        test["Vector"] = test_vectors.values
    train["Vector"] = train_vectors.values

def wait_for_file(file_path):
    print("Waiting for file")
    while not os.path.exists(file_path):
        time.sleep(1)

    if os.path.isfile(file_path):
        print("Found")
        T = pd.read_csv(file_path)
        T = T.iloc[:, :-1]
        T['Vector'] = T.values.tolist()
        os.remove(file_path)
        return T["Vector"]
    else:
        raise ValueError("%s isn't a file!" % file_path)
