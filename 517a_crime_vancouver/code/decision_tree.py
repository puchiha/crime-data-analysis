from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from common import *

def get_crime_data():
	#file = 'raw_data/crime_processed_neighbourhood.csv'
	file='raw_data/svd2.csv'
	try:
		df = pd.read_csv(file)
	except:
		exit( 'no training data' )

	return df


def d_tree():
	df = get_crime_data()

	#features = list(df.columns[:9])
	y = df["CLASSIFICATION"]
	X = df.drop(['CLASSIFICATION'], axis=1)

	xTr,xTe,yTr,yTe = train_test_split(X,y,test_size=0.4,random_state=0)


	dt = DecisionTreeClassifier(min_samples_split = 20, random_state = 99)
	tree = dt.fit(xTr, yTr)
	predictions = tree.predict(xTe)

	print "Train Accuracy :: ", accuracy_score(yTr, tree.predict(xTr))
	print "Test Accuracy  :: ", accuracy_score(yTe, predictions)

	print("-- 10-fold cross-validation --")
	cv_dt = cross_val_score(dt, xTe, yTe, cv=10)

	print("mean: {:.3f} (std: {:.3f})".format(cv_dt.mean(),
                                          cv_dt.std()))

	# Train Accuracy ::  0.9437
	# Test Accuracy  ::  0.87695
	# -- 10-fold cross-validation --
	# mean: 0.876 (std: 0.007)
	#
 	# For svd5:
	# Train Accuracy ::  0.8775
	# Test Accuracy  ::  0.6949
	# -- 10-fold cross-validation --
	# mean: 0.708 (std: 0.007)

	#for svd2:
	# Train Accuracy ::  0.860666666667
	# Test Accuracy  ::  0.69585
	# -- 10-fold cross-validation --
	# mean: 0.692 (std: 0.013)





if __name__ == "__main__":
    d_tree()
