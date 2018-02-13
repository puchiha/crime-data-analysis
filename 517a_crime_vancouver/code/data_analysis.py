from common import *

def analysis():
    df = pd.read_csv("raw_data/crime.csv")
    print "The different columns in the dataset are: ", df.columns.tolist()
    print
    print 'Types of Criminal Activities Reported: ', len(set(df.TYPE.tolist()))
    print
    print 'Types: ', set(df.TYPE.tolist())
    print
    print "Types of Neighbourhood :", len(set(df.NEIGHBOURHOOD.tolist()))
    print
    print "Types :", set(df.NEIGHBOURHOOD.tolist())

if __name__ == "__main__":
    analysis()   
