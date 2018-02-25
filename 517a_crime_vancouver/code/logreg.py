from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics

k=10
kf = KFold(n_splits=k)

data = pd.read_csv("../raw_data/crime_processed.csv").as_matrix()
X = data[:, :-1]
Y = data[:, -1]

accuracy_avg = 0
auc_avg = 0
for (trainIndex, testIndex) in kf.split(data):
    logreg = LogisticRegression()
    print trainIndex
    xTr = X[trainIndex]
    yTr = Y[trainIndex]
    xTe = X[testIndex]
    yTe = Y[testIndex]
    logreg.fit(xTr, yTr)
    preds = logreg.predict(data[testIndex])
    #get classification accuracy
    accuracy = logreg.score(xTe, yTe)
    #get area under roc curve
    auc = metrics.roc_auc_score(yTe, preds)
    accuracy_avg += accuracy
    auc_avg += auc

auc_avg = auc_av/k
accuracy_avg = accuracy_avg/k
print 'Evaluated logistic regression with'+str(k)+'-fold cv'
print 'Avg accuracy:', accuracy_avg
print 'Avg auc:', auc_avg
