from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import numpy as np
k=10
kf = KFold(n_splits=k)

data = pd.read_csv("raw_data/crime_processed_neighbourhood.csv").as_matrix()
#data = data.sample(40000).as_matrix()
print len(data[0])
X = data[:, [0,1,2,3,4,5,6,7,9]]
Y = data[:, 8]


#--------------------------------USING an RBF kernel--------------------------------



clf = svm.SVC(kernel='rbf',max_iter=1000,degree=2)
print "Training SVM classifier"
print "Using a RBF kernel "

accuracy_avg = cross_val_score(clf, X, Y, cv=10)
print 'Evaluated SVM  with '+str(k)+'-fold cv'
print 'Avg accuracy:', np.mean(accuracy_avg)


xTr, xTe, yTr, yTe = train_test_split(X, Y, test_size=0.9)
clf.fit(xTr, yTr)
print "predicting:"

preds =  clf.predict(xTe)
print "ROC_AUC_SCORE: ", metrics.roc_auc_score(yTe,preds)

fpr, tpr, thresholds = metrics.roc_curve(yTe, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr) 
plt.plot([0,1], [0,1], color='grey', linestyle='--') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('SVM ROC Curve')
plt.show() 

