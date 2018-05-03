# Compare Algorithms

from common import *
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from time import time
from scipy.stats import ttest_ind
import numpy as np

data = pd.read_csv("raw_data/crime_processed_neighbourhood.csv").as_matrix()
#data = pd.read_csv("../raw_data/sv").as_matrix()
X = data[:, [0,1,2,3,4,5,6,7,9]]
Y = data[:, 8]

plotfig = True

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR ', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 2, algorithm = 'brute')))
models.append(('D-Tree', DecisionTreeClassifier(min_samples_split = 2, random_state = 99)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='sigmoid', max_iter=1000, degree=2)))
models.append(('GP', GaussianProcessClassifier(kernel = gp.kernels.ConstantKernel() + gp.kernels.Matern(length_scale=2, nu=3/2) + gp.kernels.WhiteKernel(noise_level=1))))

# evaluate each model in turn
final_results = []
names = []
times = []
cv_mean, cv_std = [], []

print 'Model:\tmean\t\t(std dev)\ttic-toc'
for name, model in models:
	results =[]
	t = []
	for i in range(10):
		tic = time()
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')#scoring='roc_auc'
		#results.append(cv_results)
		cv_mean.append(cv_results.mean())
		cv_std.append(cv_results.std())
		toc = time()
		t.append(toc-tic)

	names.append(name)
	t_mean = np.mean(t)
	times.append(t_mean)
	msg = "%s:\t%f\t(%f)\t%f s" % (name, np.mean(cv_mean), np.mean(cv_std), t_mean)
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