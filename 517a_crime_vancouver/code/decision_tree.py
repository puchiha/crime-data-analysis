from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import  cross_val_score
from common import *

def get_crime_data():
	file = 'raw_data/crime_processed_neighbourhood.csv'
	try:
		df = pd.read_csv(file)
	except:
		exit( 'no training data' )
		
	return df

def d_tree():
	df = get_crime_data()
	features = list(df.columns[:9])
	y = df["CLASSIFICATION"]
	X = df[features]
	#	x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.4,random_state=0)
	
	dt = DecisionTreeClassifier(min_samples_split = 20, random_state = 99)
	print("-- 10-fold cross-validation --")
	cv_dt = cross_val_score(dt, X, y, cv=10)
	fit_dt = dt.fit(X, y)
	print("mean: {:.3f} (std: {:.3f})".format(cv_dt.mean(),
                                          cv_dt.std()))
                                         
	
	#visualize_tree(dt, features)
	

if __name__ == "__main__":
    d_tree() 