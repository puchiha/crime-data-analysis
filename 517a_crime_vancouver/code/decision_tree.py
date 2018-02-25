from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from common import *

def get_crime_data():
	file = 'raw_data/crime_processed_neighbourhood.csv'
	try:
		df = pd.read_csv(file)
	except:
		exit( 'no training data' )
		
	return df

def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def d_tree():
	df = get_crime_data()
	features = list(df.columns[:9])
	y = df["CLASSIFICATION"]
	X = df[features]

	dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)

	w = dt.fit(X, y)
	visualize_tree(dt, features)
	print w

if __name__ == "__main__":
    d_tree() 