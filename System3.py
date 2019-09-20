# This System does:
# Memory Based Collabrative filtering using Matrix Factrization

import numpy as np
import LoadDataset as ld
import others as ot

#===================================================================================================
### COLLABRATIVE FILTERING
df = ld.df_ratings_grouped

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

### MEMORY BASED CF (user-item CF & item-item CF)
train_data_matrix = np.zeros((ld.n_users, ld.n_movies+ld.n_extra))
for line in train_data.itertuples():
	if line[0][1] < ld.n_movies:
		train_data_matrix[line[0][0]-1, line[0][1]-1] = line[1]
	else:
		train_data_matrix[line[0][0]-1, ld.n_movies+ot.old_to_new(ot.hashed, line[0][1]) ] = line[1]


test_data_matrix = np.zeros((ld.n_users, ld.n_movies+ld.n_extra))
for line in test_data.itertuples():
	if line[0][1] < ld.n_movies:
		test_data_matrix[line[0][0]-1, line[0][1]-1] = line[1]
	else:
		test_data_matrix[line[0][0]-1, ld.n_movies+ot.old_to_new(ot.hashed, line[0][1]) ] = line[1]

#===================================================================================================
### Memory based CF (Matrix Factrization)
sparcity = round(1.0 - len(ld.df_ratings)/float(ld.n_users*ld.n_movies), 3)


from scipy.sparse.linalg import svds

### get SVD components from matrix choose k
u, s, vt = svds(train_data_matrix, k=20)

sparse_matrix = np.diag(s)

pre_pred = np.dot(u, sparse_matrix)
X_pred = np.dot(pre_pred, vt)

### analysis
model_rmse = ot.rmse(X_pred, test_data_matrix)

#===================================================================================================
