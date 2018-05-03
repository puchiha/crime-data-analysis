from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from common import *

def get_crime_data():
	file = '../raw_data/crime_processed_neighbourhood.csv'
	try:
		df = pd.read_csv(file)
	except:
		exit( 'no training data' )
		
	return df


def d_tree():
	data = get_crime_data().as_matrix()
	X = data[:, [0,1,2,3,4,5,6,7,9]]
	y = data[:, 8]
	
	xTr,xTe,yTr,yTe = train_test_split(X,y,test_size=0.4,random_state=0)
	
	
	dt = DecisionTreeClassifier(min_samples_split = 20, random_state = 99)
	tree = dt.fit(xTr, yTr)
	preds = tree.predict(xTe)

	print "Train Accuracy :: ", accuracy_score(yTr, tree.predict(xTr))
	print "Test Accuracy  :: ", accuracy_score(yTe, preds)
 
	print("-- 10-fold cross-validation --")
	cv_dt = cross_val_score(dt, xTe, yTe, cv=10)
	
	print("mean: {:.3f} (std: {:.3f})".format(cv_dt.mean(),
                                          cv_dt.std()))
                                         
	'''
	Train Accuracy ::  0.9437
	Test Accuracy  ::  0.87695
	-- 10-fold cross-validation --
	mean: 0.876 (std: 0.007)
	'''
	#	---- STAT TEST ----
	from scipy.stats import ttest_ind
	results = pd.DataFrame({'preds': preds, 'yTe': yTe})

	value, pvalue = ttest_ind(preds, yTe, equal_var = True)
	print (value, pvalue)
	if pvalue >= 0.05:
		print('Dtree is a good predictor for classification')
	else:
		print('Dtree is a bad predictor for classification')

if __name__ == "__main__":
    d_tree() 