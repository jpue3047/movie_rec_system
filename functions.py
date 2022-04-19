import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_nested_value(df, field_name, nested_field, new_col):
    """ Normalize nested json column and add into existing dataframe
    Args:
        df (dataframe): dataframe with json values
        field_name (string): field name to extract in nested json
        nested_field (string): nested json field which contains values
        new_col (string): name of new column
    Returns:
        df (dataframe): dataframe with new column
    """
   
    df[field_name] = df[field_name].apply(json.loads)
    for index,i in zip(df.index, df[field_name]):
        list_values = []
        for j in range(len(i)):
            list_values.append((i[j][nested_field])) 
        df.loc[index, new_col] = str(list_values)   
    return df


def extract_value(df, field_name, nested_field, new_col):
    """ Normalize json column and add into existing dataframe
    Args:
        df (dataframe): dataframe with json values
        field_name (string): field name to extract json
        nested_field (string): json field which contains values
        new_col (string): name of new column
    Returns:
       df (dataframe): dataframe with new column
    """
    df[field_name] = df[field_name].apply(json.loads)
    for index,i in zip(df.index, df[field_name]):
       
        for j in range(len(i)):
            if (i[j]['job'] == nested_field):
                df.loc[index, new_col] = str([i[j]['name']])         
    return df


def combined_attributes( row, c_list):
    """ Concatenate list of columns
    Args:
        row(string): row
        c_list (list): list of columns to combined
    Returns:
        x(string) : column aggregated 
    """
    x = f''
    for i in c_list:
        x += str(row[i]).replace('[','').replace(']','').replace(',',' ')+' '
    return x


def get_cosine(df, col_name, col_id):
    """ Calculate cosine similarity 
    Args:
        df (dataframe): dataframe with movies and attributes
        col_name (string): column which contains movie attributes
        col_id (string): column name with id movies
    Returns:
        sim_df(dataframe): dataframe showing cosine similarity for all movies
    """
    cv = CountVectorizer()
    cosine_matrix = cv.fit_transform(df[col_name])
    features = cv.get_feature_names()  
    cosine_sim = cosine_similarity(cosine_matrix)
    sim_df = pd.DataFrame(cosine_sim, index =df[col_id] ,  columns = df[col_id])

    return sim_df


def top_items(df, id, top_n, dict):
    """ Recomendation top movies based on movie input
    Args:
        df (dataframe): cosine similary dataframe
        id (int): movie target to make recomendations
        top_n (int): number of movies to make recomendation
        dict (dictionary): dictionary with movie id and movie title
    Returns:
        string: print top_n recomendations
    """
    df_value = df[[id]]
    df_value = df_value[df_value.index!=id]
    df_sorted = df_value.sort_values( id,ascending =False).head(top_n)
    lst_id = df_sorted.index.tolist()
    movie_list = []
    for i in lst_id:
        movie_list.append(dict[i])
    movie_rec = dict[id]
    return print(f'''\n Movies Recomendation for {movie_rec}:''', movie_list)

