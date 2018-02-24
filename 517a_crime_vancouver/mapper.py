from common import *


def _mapper(df, index_col, count_name = "COUNT"):

    code = df.groupby(index_col).first().sort_index()
    code.loc[:,count_name] = df.groupby(index_col).size()
    code = code.reset_index().loc[:,index_col + ["COUNT"]]
    code = code.sort_values(index_col).reset_index(drop=True)
    index_array = [df[i].as_matrix() for i in index_col]
    factored = pd.factorize( pd.lib.fast_zip( index_array ), sort=True)[0]
    s = pd.Series(factored, index=df.index)
    return s, code


index_column = ['NEIGHBOURHOOD']

data = pd.read_csv("raw_data/crime_processed.csv")
df,code = _mapper(data, index_column)
data = pd.concat([data,df], axis = 1)

#data = data.set_index('TYPE')

data = data.drop(labels = ['NEIGHBOURHOOD'], axis=1)

data = data.rename(columns = {0 : 'NEIGHBOURHOOD'})

data.to_csv("raw_data/crime_processed_neighbourhood.csv", index_label = False)
code.to_csv("raw_data/crime_neighbourhood_codes.csv", index_label = False)


