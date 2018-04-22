from common import *

def analysis():
    file_path =  '/Users/puchiha/g_drive/SP18/CSE517A/data/crime_data/'
    df = pd.read_csv(file_path + r"crime_processed.csv")
    print "The different columns in the dataset are: ", df.columns.tolist()
    print
    print 'Types of Criminal Activities Reported: ', len(set(df.TYPE.tolist()))
    print
    print 'Various Types of Criminal Activities Reported: '
    for i in set(df.TYPE.tolist()): print i
    print
    print "Types of Neighbourhood :", len(set(df.NEIGHBOURHOOD.tolist()))
    print
    print "Various Types of Neighbourhood :"
    for i in set(df.NEIGHBOURHOOD.tolist()): print i
    print
    print "Per Type count values:"
    print df['TYPE'].value_counts().sort_index()

# Distribution of crimes per day
'''
def plot_crimes_per_day():
    df = pd.read_csv("raw_data/crime_processed.csv")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(20,10))
    plt.tick_params(labelsize=16)
    plt.title('Distribution of Crimes per day', fontsize=20)
    plt.xlabel('Crimes Per Day')
    plt.ylabel('Frequency')
    sns.distplot(df.resample('D').size(), bins = 50)
'''    
if __name__ == "__main__":
    analysis()   
