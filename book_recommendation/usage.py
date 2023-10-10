import numpy as np
from joblib import load

# Popularity Based
top_50_df = load("top_50_df.joblib")

print(top_50_df['Book-Title'])

# Collaborative Filtering Based
similarities = load("recommendations.joblib")
pt = load("pt.joblib")

def recommend(book_title):
    try:
        book_index = np.where(pt.index==book_title)[0][0]
        distances = similarities[book_index]
        book_objs = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
        return [pt.index[i[0]] for i in book_objs]

    except:
        return []

print(recommend("The Notebook"))