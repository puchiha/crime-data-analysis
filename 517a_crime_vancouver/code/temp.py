
'''
Train and run Gaussian Processes. Evaluate and compare the predictions using at least two differnt kernels via 10-fold cross-validation with a suitable error measure (we recommend negative log predictiv density as it takes the predictive uncertainty into account).

Prakrit Shrestha | 2018
Crime Data Analysis - Running a basic GP


'''
from common import *
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel


def get_crime_data():
	file = '../raw_data/crime_processed_neighbourhood.csv'
	try:
		df = pd.read_csv(file)
	except:
		exit( 'no training data' )
		
	return df


def g_process():
	#data = get_crime_data().as_matrix()
	#features = list(data.columns[:9])
	#X = data[:, [0,1,2,3,4,5,6,7,9]]
	#y = data[:, 8]

	dToy = get_crime_data().head(10).as_matrix()
	X = dToy[:, [0,1,2,3,4,5,6,7,9]]
	y = dToy[:, 8]
	X = X[:, 5].reshape(-1,1)
	kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
	#kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))


	print 'running GP\n\n'

	gp = GaussianProcessRegressor(alpha=1e-10, copy_X_train=True,
		kernel=kernel, n_restarts_optimizer=0, normalize_y=False,
		optimizer='fmin_l_bfgs_b', random_state=None)
	
	accuracy_avg = cross_val_score(gp, X, y, cv=10)

	#	log_lik, log_grad =  gp.log_marginal_likelihood(theta=None, eval_gradient=False)
	xTr, xTe, yTr, yTe = train_test_split(X, y, test_size=0.1)
	gp.fit(xTr, yTr)

	y_pred, sigma = gp.predict(xTr, return_std=True)
	#print y, y_pred, '\n -- \n'
	#print sigma 
	acc = metrics.accuracy_score(yTr, y_pred)
	print ("Accuracy: {0:.2f}%".format(acc * 100.0))
	print ('PLOTTING')
	fig = plt.figure()
	plt.plot(xTr, yTr, 'r.', markersize=5, label=u'Observations')
	plt.plot(xTr, y_pred, 'b:', label=u'Prediction')
	plt.fill(np.concatenate([xTr, xTr[::-1]]),
	         np.concatenate([y_pred - 1.9600 * sigma,
	                        (y_pred + 1.9600 * sigma)[::-1]]),
	         alpha=.5, fc='g', ec='None', label='95% confidence interval')
	plt.show()

	

if __name__ == "__main__":
    g_process() 
