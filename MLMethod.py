from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats
import FuzzyRough as fr
import FRNN
import pandas as pd


class MLMethod:
    def __init__(self, name):
        self.name = name

    def result(self, test, train):
        raise NotImplementedError


class SklearnMethods(MLMethod):
    def __init__(self, name, model):
        super(SklearnMethods, self).__init__(name)
        self.model = model

    def result(self, test, train):
        x_training, y_training = pd.DataFrame(item for item in train["Vector"]),  pd.Series(train["Label"], dtype=int)
        x_test, y_test = pd.DataFrame(item for item in test["Vector"]), pd.Series(test["Label"], dtype=int)
        self.model.fit(x_training, y_training)
        y_predicted = self.model.predict(x_test)
        return scipy.stats.pearsonr(y_test, y_predicted)


class SVM(SklearnMethods):
    def __init__(self):
        super(SVM, self).__init__("Support Vector Machine", svm.LinearSVC(max_iter=1200000))


class RandomForest(SklearnMethods):
    def __init__(self):
        super(RandomForest, self).__init__("Random Forests", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0))


class NearestNeighbour(SklearnMethods):
    def __init__(self, k):
        super(NearestNeighbour, self).__init__("KNN", KNeighborsClassifier(n_neighbors=k))


class FuzzyRoughNN(MLMethod):
    def __init__(self, name, model, relation, distance):
        super(FuzzyRoughNN, self).__init__(name)
        self.model = model
        self.relation = (relation, distance)

    def result(self, test, train):
        raise NotImplementedError


class FRNN_OWA(FuzzyRoughNN):
    def __init__(self, name, relation, implicator, t_norm,  weights_lower, weights_upper, k, distance=False):
        super(FRNN_OWA, self).__init__(
            name,
            lambda df, sample, r: FRNN.frnn_owa(df, sample, k, r, implicator, t_norm, weights_lower, weights_upper),
            relation,
            distance
        )

    def result(self, test, train):
        if self.relation[1]:
            relation = fr.relation_from_d_measure(train["Vector"], self.relation[0])
        else:
            relation = self.relation[0]
        y_predicted = []
        for index, row in test.iterrows():
            y_predicted.append(self.model(train, row, relation))
        y_test = pd.Series(test["Label"], dtype=int)
        return scipy.stats.pearsonr(y_test, y_predicted)
