from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics

k=10
kf = KFold(n_splits=k)

data = pd.read_csv("../raw_data/crime_processed_neighbourhood.csv").as_matrix()
X = data[:, :-1]
Y = data[:, -1]

accuracy_avg = 0
for (trainIndex, testIndex) in kf.split(data):
    logreg = LogisticRegression()
    xTr = X[trainIndex]
    yTr = Y[trainIndex]
    xTe = X[testIndex]
    yTe = Y[testIndex]
    logreg.fit(xTr, yTr)
    preds = logreg.predict(xTe)
    #get classification accuracy
    accuracy = logreg.score(xTe, yTe)
    accuracy_avg += accuracy

accuracy_avg = accuracy_avg/k
print 'Evaluated logistic regression with '+str(k)+'-fold cv'
print 'Avg accuracy:', accuracy_avg
