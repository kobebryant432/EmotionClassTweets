##
#Imports
from frlearn.utils.owa_operators import strict, additive, invadd
from frlearn.ensembles.classifiers import FRNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import frlearn.base as b
import scipy.stats

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

f = FRNN(upper_weights=additive(), upper_k=2, lower_weights=additive(), lower_k=2)
con = f.construct(X, y)
scores = con.query(X)
b.select_class(scores, labels=con.classes)

X_test = X[-5:]
X_train = X[:-5]
y_test = y[-5:]
y_train = y[:-5]

results = []
for k in [1, 2, 3, 5, 10, 15, 20]:
    clas = b.FitPredictClassifier(FRNN, upper_weights=additive(), upper_k=k, lower_weights=additive(), lower_k=k)
    clas.fit(X_train, y_train)
    y_pred = clas.predict(X_test)
    print(y_pred, y_test)

print("next")
for k in [1, 2, 3, 5, 10, 15, 20]:
    clas = KNeighborsClassifier(n_neighbors=k,)
    clas.fit(X_train, y_train)
    y_pred = clas.predict(X_test)
    print(y_pred, y_test)
