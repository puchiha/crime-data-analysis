# Compare Algorithms

from common import *
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from time import time
from scipy.stats import ttest_ind


data = pd.read_csv("../raw_data/nn_processed.csv").as_matrix()
X = data[:, [0,1,2,3,4,5,6,7,9]]
Y = data[:, 8]

plotfig = False

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR ', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('GP', GaussianProcessClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('D-Tree', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
times = []
scoring = 'accuracy'
print 'Model:\tmean\t\t(std dev)\ttic-toc'
for name, model in models:
	tic = time()
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	toc = time()
	times.append(toc-tic)
	msg = "%s:\t%f\t(%f)\t%f ms" % (name, cv_results.mean(), cv_results.std(), toc - tic)
	print(msg)

# boxplot algorithm comparison
if plotfig:
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	#plt.show()
	plt.savefig('compare_results.png')