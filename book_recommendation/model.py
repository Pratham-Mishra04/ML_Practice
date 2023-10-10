import pandas as pd
import numpy as np

ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")
books = pd.read_csv("Books.csv")

book_ratings=ratings.merge(books, on='ISBN')

# Popularity Based Recommendation Model

num_rating_df = book_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={"Book-Rating":"num_ratings"}, inplace=True)

avg_rating_df = book_ratings.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={"Book-Rating": "avg_ratings"}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings']>250].sort_values('avg_ratings', ascending=False)

popular_df=popular_df.merge(books, on="Book-Title").drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_ratings']]
top_50_df = popular_df.head(50)

# Collaborative Filtering Based Recommendation Model

users_200ratings_series=book_ratings.groupby('User-ID').count()['Book-Rating'] > 200
users_200ratings_ids = users_200ratings_series[users_200ratings_series].index
filtered_ratings = book_ratings[book_ratings['User-ID'].isin(users_200ratings_ids)]

books_50ratings_series=filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
books_50ratings_titles = books_50ratings_series[books_50ratings_series].index
filtered_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(books_50ratings_titles)]

pt = filtered_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(pt)

from joblib import dump
dump(similarities, 'recommendations.joblib')
dump(pt, 'pt.joblib') 
dump(top_50_df, 'top_50_df.joblib')