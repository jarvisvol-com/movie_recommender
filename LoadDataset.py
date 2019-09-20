import numpy as np
import pandas as pd

#===================================================================================================
tags = pd.read_csv(r'../Dataset/tags.csv')
df_tags = tags[['userId', 'movieId', 'tag']]

ratings = pd.read_csv(r'../Dataset/ratings.csv')
df_ratings = ratings.drop('timestamp', axis=1)

df_ratings_grouped = df_ratings.groupby(['userId', 'movieId']).aggregate(np.max)
### we look for the user who has rated the same movie twice and take the max rating

#===================================================================================================
movie_list = pd.read_csv(r'../Dataset/movies.csv')

genres = pd.get_dummies(movie_list['genres'])

df_movie = pd.concat([movie_list, genres], axis=1)

rating_avg_count = pd.DataFrame(df_ratings.groupby('movieId')['rating'].agg(['mean','count']))

#===================================================================================================

n_users = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
n_total_movies = df_movie['movieId'].nunique()

rated_movies = df_ratings['movieId'].unique()
count = 0
for i in rated_movies:
	if i > 14026:
		count+=1

n_extra = count
#===================================================================================================
