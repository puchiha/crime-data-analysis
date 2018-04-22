import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import TruncatedSVD #does not recenter data

raw_data = pd.read_csv("raw_data/crime_processed_neighbourhood.csv")
#exclude classification column
data = raw_data.drop('CLASSIFICATION', axis=1)


# s = np.linalg.svd(data, compute_uv=False)
# print s

#fit svd transform for desired number of dims
np.random.seed(9)
num_dim=1
svd = TruncatedSVD(num_dim)
svd.fit(data)
#print svd.singular_values_/np.sum(svd.singular_values_)

#transform data, set column names, save
new_data = svd.transform(data)
new_data = pd.DataFrame(new_data)
#add classification column back in
new_data = new_data.reset_index(drop=True)
raw_data = raw_data.reset_index(drop=True)
new_data['CLASSIFICATION'] = raw_data['CLASSIFICATION']

new_data.to_csv('raw_data/svd'+str(num_dim)+'.csv', index=False)

#visualize
# plt.figure()
# plt.scatter(new_data[0], new_data[1], c=new_data['CLASSIFICATION'], cmap='coolwarm')
# plt.title('SVD transform with '+str(num_dim)+' dimensions')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.show()

# plt.figure()
# pd.plotting.parallel_coordinates(new_data, 'CLASSIFICATION')
# plt.show()
