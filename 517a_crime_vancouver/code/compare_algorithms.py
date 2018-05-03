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

# load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
data = pd.read_csv("raw_data/nn_processed.csv").as_matrix()
#data = data.sample(40000).as_matrix()
#print len(data[0])
X = data[:, [0,1,2,3,4,5,6,7,9]]
Y = data[:, 8]


# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('GP', GaussianProcessClassifier(	kernel = gp.kernels.ConstantKernel() + gp.kernels.Matern(length_scale=2, nu=3/2) + gp.kernels.WhiteKernel(noise_level=1)
)))
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
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()
plt.savefig('compare_results.png')