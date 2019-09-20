import pandas as pd
import LoadDataset as ld

#===================================================================================================
# to get the best movies by genre
def best_movies_by_genre(movie_score, genre ,n):
	res = pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['Weighted score'],
						ascending=False)[['title', 'Weighted score']][:n])
	return res

#===================================================================================================
# to return correlation of one movie with all others in decending order
def corr(table, df, count=ld.rating_avg_count['count']):
	corr_with = pd.DataFrame(table.corrwith(df), columns=['Corr'])
	corr_with['n_ratings'] = count
	corr_with = corr_with[corr_with['n_ratings'] > 100].sort_values('Corr', ascending=False)
	return corr_with

#===================================================================================================
# give the title of whose id in given
def id_to_title(movieId, df = ld.df_movie):
	res = df[df['movieId'] == movieId]['title']
	return res

def title_to_id(title, df = ld.df_movie):
	res = df[df['title'] == title]['movieId']
	return res

#===================================================================================================
# System 3 & 4
hashed = pd.DataFrame(columns=['movieId'])
val = 0
for i in ld.rated_movies:
	if i > ld.n_movies:
		hashed.loc[val] = i
		val = val+1

def old_to_new(hashed, val):
	res = hashed[hashed['movieId'] == val].index
	return res

#===================================================================================================
# System 3
import math
from sklearn.metrics import mean_squared_error

def rmse(predictions, ground_truth):
	predictions = predictions[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return math.sqrt(mean_squared_error(predictions, ground_truth))

#===================================================================================================
