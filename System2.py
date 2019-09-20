# This System does :
# 1) User-Item Memory based Collabrative filtering
# 2) Item-Item Memory based Collabrative filtering
#   (Uses Cosine Similarity)

import numpy as np
import LoadDataset as ld
import Others as ot

#===================================================================================================
### COLLABRATIVE FILTERING
df = ld.df_ratings_grouped

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

#===================================================================================================
### MEMORY BASED CF (user-item CF & item-item CF)

### Creating two User-Item matrix
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
### Calculating the cosine similarity
from sklearn.metrics.pairwise import pairwise_distances

user_simi = pairwise_distances(train_data_matrix, metric='cosine')
item_simi = pairwise_distances(train_data_matrix.T, metric='cosine')

#===================================================================================================
def itemPredict(ratings, similarity = item_simi):

	denominator = np.array([np.abs(similarity).sum(axis=1)])

	numerator = ratings.dot(similarity)

	result = numerator/denominator
	return result

#===================================================================================================
def userPredict(ratings, similarity = user_simi):

	denominator = np.array([np.abs(similarity).sum(axis=1)])

	mean_user_rating = ratings.mean(axis=1) ### the mean user ratings
	ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) ### ratings diffrence
	numerator = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)

	result = numerator/denominator.T
	return result

#===================================================================================================
### Getting the best predictions for the item & user similarity
item_predictions = itemPredict(train_data_matrix, item_simi)
user_predictions = userPredict(train_data_matrix, user_simi)

### Analysis of the rmse
item_rmse = ot.rmse(item_predictions, test_data_matrix)
user_rmse = ot.rmse(user_predictions, test_data_matrix)

#===================================================================================================

