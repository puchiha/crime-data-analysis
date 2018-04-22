from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

k=10
kf = KFold(n_splits=k)

data = pd.read_csv("raw-data/crime_processed_neighbourhood.csv").as_matrix()
X = data[:, [0,1,2,3,4,5,6,7,9]]
Y = data[:, 8]

logreg = LogisticRegression()
accuracy_avg = cross_val_score(logreg, X, Y, cv=10)

print 'Evaluated logistic regression with '+str(k)+'-fold cv'
print 'Avg accuracy:', np.mean(accuracy_avg)

#plot roc curve
xTr, xTe, yTr, yTe = train_test_split(X, Y, test_size=0.1)
preds = logreg.fit(xTr, yTr).decision_function(xTe)
fpr, tpr, thresholds = metrics.roc_curve(yTe, preds)
roc_auc = metrics.auc(fpr, tpr)
print "Area under roc curve:", roc_auc
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

'''
Avg accuracy: 0.601822160415
Area under roc curve: 0.70335792905
'''