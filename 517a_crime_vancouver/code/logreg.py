from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics

k=10
kf = KFold(n_splits=k)
#TODO: load dataset

xIndex = np.arange(data.shape[1]-1)
yIndex = -1

accuracy_avg = 0
auc_avg = 0
for (trainIndex, testIndex) in kf.split(data):
    logreg = LogisticRegression
    #TODO: set X and Y indices
    xTr = data[trainIndex][:,xIndex]
    yTr = data[trainIndex][:,yIndex]
    xTe = data[testIndex][:,xIndex]
    yTe = data[testIndex][:,yIndex]
    logreg.fit(data[trainIndex][X], data[testIndex][Y])
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
