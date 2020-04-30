##
#Imports
from frlearn.utils.owa_operators import strict, additive, invadd
from frlearn.ensembles.classifiers import FRNN
from sklearn import datasets
import frlearn.base as b

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

f = FRNN(upper_weights=additive(), upper_k=2, lower_weights=additive(), lower_k=2)
con = f.construct(X, y)
scores = con.query(X)
b.select_class(scores, labels=con.classes)


clas = b.FitPredictClassifier(FRNN, upper_weights=additive(), upper_k=0, lower_weights=strict(), lower_k=0)
clas.fit(X, y)
clas.predict(X)