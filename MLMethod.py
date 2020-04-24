from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats
from frlearn.neighbours.classification import FRNNClassifier
from frlearn.neighbours.neighbour_search import KDTree

import pandas as pd

class MLMethod:
    """
        A class used to represent a machine learning method

        ...

        Attributes
        ----------
        name : str
           the name of the embedding method

        Methods
        -------
        result(test, train):
            Performs the classifiction task on the test set given the training and reports the pearson correlation
            between the predicted label and actual label of the test samples.
        """
    def __init__(self, name):
        self.name = name

    def result(self, test, train):
        raise NotImplementedError


class SklearnMethods(MLMethod):
    """
        A class extending MLMethod used to represent a machine learning method from the sklearn package.
        It has subclasses SVM, RandomForest and NearestNeighbour
        ...
        Attributes
        ----------
        name : super
        model :
            The model of the sklearn machine learning method, must have all sklearn functionality

        Methods
        -------
        result(test, train):
            Performs the classifiction task on the test set given the training and reports the pearson correlation
            between the predicted label and actual label of the test samples.
        """
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def result(self, test, train):
        x_training, y_training = pd.DataFrame(item for item in train["Vector"]),  pd.Series(train["Label"], dtype=int)
        x_test, y_test = pd.DataFrame(item for item in test["Vector"]), pd.Series(test["Label"], dtype=int)
        self.model.fit(x_training, y_training)
        y_predicted = self.model.predict(x_test)
        return scipy.stats.pearsonr(y_test, y_predicted)


class SVM(SklearnMethods):
    """
        A class extending SklearnMethods used to represent a linear support vector machine.
    """
    def __init__(self):
        super().__init__("Support Vector Machine", svm.LinearSVC(max_iter=1200000))


class RandomForest(SklearnMethods):
    """
        A class extending SklearnMethods used to represent a Random Forest.
    """
    def __init__(self):
        super().__init__("Random Forests", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0))


class NearestNeighbour(SklearnMethods):
    """
        A class extending SklearnMethods used to represent a Random Forest. Input k, the number of nearest neighbours.
    """
    def __init__(self, k):
        super().__init__("KNN", KNeighborsClassifier(n_neighbors=k))


class FRNN_OWA(SklearnMethods):
    """
        A class extending FuzzyRoughNN used to represent a OWA type FRNN.
        Input: relation, the fuzzy tolerance relation/distance measure
               implicator, the implicator
               t_norm, the t_norm
               weights_lower, the weighting scheme of the lower approximation
               weights_upper, the weighting scheme of the upper approximation
               k, the number of nearest neighbours
               distance, true is relation is a distance measure.
    """
    def __init__(self, name, weights_lower, weights_upper):
        search = KDTree()
        super().__init__(
            name,
            model=FRNNClassifier(nn_search=search, upper_weights=weights_upper)
        )

    def result(self, test, train):
        x_training, y_training = pd.DataFrame(item for item in train["Vector"]),  pd.Series(train["Label"], dtype=int)
        x_test, y_test = pd.DataFrame(item for item in test["Vector"]), pd.Series(test["Label"], dtype=int)
        self.model.fit(x_training, y_training)
        y_predicted = self.model.predict(x_test)
        return scipy.stats.pearsonr(y_test, y_predicted)
