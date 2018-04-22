from common import *

import os
from scipy.sparse import csr_matrix
import itertools


def _as_list(x):
    return x if type(x) == list else [x]

def _strip(df,columns):
    for c in _as_list(columns):
        df.loc[:,c] = df[c].str.strip()
    return df

def _lower(df,columns):
    for c in _as_list(columns):
        df.loc[:,c] = df[c].str.lower()
    return df

def category(type):
    if 'Collision' in type:
        return 1
    else:
        return -1 

def time_category(x, type):
    x = x.split(':')
    if type == 'hour':
        return x[0]
    elif type == 'minute':
        return x[1]
        
def pack():
    df = pd.read_csv("raw_data/crime.csv")
    df = df.dropna(subset = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE'])
    df['DATE'] = df.YEAR.astype('str').map(str) + '/' + df.MONTH.astype('str') + '/' +df.DAY.astype('str') + ' ' + df.HOUR.astype('str') + ':' + df.MINUTE.astype('str')
    df.DATE = df.DATE.apply(pd.to_datetime).dt.date
    df['TIME'] = df.HOUR.astype('int64').astype('str') + ':' + df.MINUTE.astype('int64').astype('str')
    df['TIME_HOUR'] = df.TIME.map(lambda x: time_category(x, 'hour'))
    df['TIME_MINUTE'] = df.TIME.map(lambda x: time_category(x, 'minute'))
    df = df.dropna(subset = ['NEIGHBOURHOOD'])
    df.TYPE = df.TYPE.astype('str')
    df.NEIGHBOURHOOD = df.NEIGHBOURHOOD.astype('str')
    df['DAY_OF_WEEK'] = pd.DatetimeIndex(df['DATE']).dayofweek
    df['CLASSIFICATION'] = df.TYPE.apply(category)
    df = df.rename(columns = {'DAY': 'DAY_OF_MONTH'})
    df = df.drop(labels=['DATE','HOUR', 'MINUTE', 'HUNDRED_BLOCK', 'X', 'Y', 'TYPE', 'TIME'], axis = 1)
    class1 = df[df.CLASSIFICATION == 1]
    class2 = df[df.CLASSIFICATION == -1]
    class2 = class2.sample(27859)
    df = pd.concat([class1, class2])
    df.to_csv('raw_data/crime_processed.csv', index_label = False)

if __name__ == "__main__":
    pack()
