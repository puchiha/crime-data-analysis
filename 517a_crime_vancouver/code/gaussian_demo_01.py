
'''
Train and run Gaussian Processes. Evaluate and compare the predictions using at least two differnt kernels via 10-fold cross-validation with a suitable error measure (we recommend negative log predictiv density as it takes the predictive uncertainty into account).

Prakrit Shrestha | 2018
Crime Data Analysis - Running a basic GP


'''
from common import *
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import sklearn.gaussian_process as gp
#	from sklearn.gaussian_process import kernels, GaussianProcessRegressor, GaussianProcessClassifier
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
#	from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.utils import shuffle

def get_crime_data():
	file = '../raw_data/crime_processed_neighbourhood.csv'
	try:
		df = pd.read_csv(file)
	except:
		exit( 'no training data' )
		
	return shuffle(df)


def g_process():
	# data = get_crime_data().as_matrix()
	# X = data[:, [1,2,3,4,5,6,7,9]]
	# y = data[:,8]
	dToy = get_crime_data().head(100).as_matrix()
	X = dToy[:, [1,2,3,4,5,6,7,9]]
	X = dToy[:, [3,4]]
	y = dToy[:, 8]

	xTr, xTe, yTr, yTe = train_test_split(X, y, test_size=0.15)
	#kernel = gp.kernels.RBF(np.ones((X.shape[1], 1))) \
	#    * gp.kernels.ConstantKernel() \
	#    + gp.kernels.WhiteKernel()	

	kernel = gp.kernels.ConstantKernel() + gp.kernels.Matern(length_scale=2, nu=3/2) + gp.kernels.WhiteKernel(noise_level=1)

	clf = gp.GaussianProcessClassifier(kernel, n_restarts_optimizer=0)
	print("Fitting Gaussian Process on input of shape {0}...".format(xTr.shape))
	clf.fit(xTr, yTr)
	print("Learned kernel: {0}".format(str(clf.kernel_)))

	y_pred = clf.predict(xTe)
	print y_pred
	print yTe

	acc = metrics.accuracy_score(yTe, y_pred)
	#	acc  = cross_val_score(clf, xTe, yTe, cv=10)

	print("Accuracy: {0:.2f}%".format(acc * 100.0))

	f0_min = X[:, 0].min() - 1
	f0_max = X[:, 0].max() + 1
	f1_min = X[:, 1].min() - 1
	f1_max = X[:, 1].max() + 1
	step = 0.2
	f0, f1 = np.meshgrid(np.arange(f0_min, f0_max, step),
                         np.arange(f1_min, f1_max, step))
	grid_data = np.c_[f0.ravel(), f1.ravel()]
	print(X.shape)
	print(xTr.shape)
	print(grid_data.shape)
	prob_grid = clf.predict_proba(grid_data)

	#prob_grid = prob_grid.reshape((f0.shape[0], f0.shape[1],-1))
	
	print prob_grid.shape, '\n --- \n' ,prob_grid.squeeze().shape
	
	plt.figure(figsize=(6, 6))
	plt.imshow(prob_grid.squeeze(), extent=(f0_min, f0_max, f1_min, f1_max),
	           origin='lower')


	plt.scatter(X[y==-1, 0], X[y==-1, 1], s=30, c='red', edgecolors='black')
	plt.scatter(X[y==1, 0], X[y==1, 1], s=30, c='green', edgecolors='black')
	plt.show()

if __name__ == "__main__":
    g_process() 
