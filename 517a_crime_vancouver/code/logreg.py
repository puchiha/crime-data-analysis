from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score

k=10
kf = KFold(n_splits=k)

data = pd.read_csv("../raw_data/crime_processed_neighbourhood.csv").as_matrix()
X = data[:, [0,1,2,3,4,5,6,7,8,10]]
Y = data[:, 9]

logreg = LogisticRegression()
accuracy_avg = cross_val_score(logreg, X, Y, cv=10)

print 'Evaluated logistic regression with '+str(k)+'-fold cv'
print 'Avg accuracy:', accuracy_avg

# accuracy_avg = 0
# for (trainIndex, testIndex) in kf.split(data):
#     logreg = LogisticRegression()
#     xTr = X[trainIndex]
#     yTr = Y[trainIndex]
#     xTe = X[testIndex]
#     yTe = Y[testIndex]
#     logreg.fit(xTr, yTr)
#     preds = logreg.predict(xTe)
#     #get classification accuracy
#     accuracy = logreg.score(xTe, yTe)
#     accuracy_avg += accuracy
#
# accuracy_avg = accuracy_avg/k
