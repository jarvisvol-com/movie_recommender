# This Syetem does :
# 1) Calculate Movie Score and give weighted rating as per no of reviews.
# 2) Give Correlation of a selected movies with other movies.

import pandas as pd
import Others as ot
import LoadDataset as ld

#===================================================================================================
# clearing movie with less count of rating and getting a movie score
min_r = 40  # minimum number of reviews required to be considered in movie score

movie_score = ld.rating_avg_count.loc[ld.rating_avg_count['count']>min_r]

#===================================================================================================
mean_all = ld.df_ratings['rating'].mean()

def weighted_rating(x, m=min_r, A=mean_all):
	v = x['count']
	R = x['mean']
	# Calculation based on the IMDB formula
	return (v/(v+m) * R) + (m/(m+v) * A)

# applying the weighted score to movie
movie_score['Weighted score'] = movie_score.apply(weighted_rating, axis=1)

#===================================================================================================
# Creating some another fresh and sorted data frames for better work
df = pd.merge(movie_score, ld.df_movie, on = 'movieId')

df_sub = df[['movieId', 'Weighted score', 'title']]
df_sub = df_sub.sort_values(['Weighted score'], ascending = False)

# TO GET THE BEST MOVIES BY GENRE
# ----> best_movies = ot.best_movies_by_genre(df, 'Action' , 10)
#                                                  #genre    #no of movies

#===================================================================================================
# Pivot Table
table_ratings = pd.pivot_table(ld.df_ratings, index='userId', columns='movieId', values='rating')

# TO GET THE CORRELATION OF A MOVIE WITH OTHER MOVIES
# -----> movie_corr = ot.corr(table_ratings, table_ratings[4])
#                                          #table_ratings[movieId]

#===================================================================================================
