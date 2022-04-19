import pandas as pd
import functions
import sys

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#load csv files
credit_df = pd.read_csv('tmdb_5000_credits.csv')
movies_df = pd.read_csv('tmdb_5000_movies.csv')
movies_df.rename(columns = {'id':'movie_id'} , inplace = True)

#normalize json values
credit_df = functions.extract_nested_value( credit_df, 'cast', 'name', 'movie_cast')
movies_df = functions.extract_nested_value( movies_df, 'genres', 'name', 'movie_genre')
movies_df = functions.extract_nested_value( movies_df, 'keywords', 'name', 'movie_keyword')
credit_df = functions.extract_value( credit_df, 'crew', 'Director', 'movie_director')

#merge dataframes
data_df = pd.merge(movies_df[['movie_id','title','movie_genre','movie_keyword']], credit_df[['movie_id', 'movie_cast', 'movie_director']], on ='movie_id')


#combine attributes to use for cosine similarity
lst_attr = ['movie_genre','movie_keyword', 'movie_cast','movie_director']

data_df['all_attr'] = data_df.apply(lambda row: functions.combined_attributes(row,lst_attr), axis =1)

#create dataframe to calculate cosine similarity
df_final = data_df[['movie_id', 'title', 'all_attr']].copy()

#create dict of movie name and id
movie_master = dict(zip(df_final['movie_id'], df_final['title']))

#calculate cosine similarity
sim_df = functions.get_cosine( df_final, 'all_attr', 'movie_id')

#recomendation
listn = functions.top_items(sim_df, 19995, 10, movie_master)















