import pandas as pd

movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

movies=movies.merge(credits, on="title")
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date']]
movies.dropna(inplace=True)

import ast

def fetch_from_obj(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'].replace(" ",""))
    return L

def fetch_from_obj_first3(obj):
    count = 0
    L = []
    for i in ast.literal_eval(obj):
        if count!=3:
            L.append(i['name'].replace(" ",""))
            count+=1
        else:
            break
    return L

def fetch_from_obj_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'].replace(" ",""))
            break
    return L

movies['genres']=movies['genres'].apply(fetch_from_obj)
movies['keywords']=movies['keywords'].apply(fetch_from_obj)
movies['cast']=movies['cast'].apply(fetch_from_obj_first3)
movies['crew']=movies['crew'].apply(fetch_from_obj_director)
movies['release_date']=movies['release_date'].apply(lambda x:["ReleaseYear"+x.split("-")[0]])
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']+movies['release_date']

df = movies[['id', 'title', 'tags']]

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def stem(x):
    L = []

    for i in x:
        L.append(ps.stem(i.lower()))
    return " ".join(L)

df.loc[:,'tags']=df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

similarities = cosine_similarity(vectors)

from joblib import dump
dump(similarities, 'recommendations.joblib')
dump(df, 'df.joblib') 