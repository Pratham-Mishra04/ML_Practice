from joblib import load

similarities = load("recommendations.joblib")
df = load("df.joblib")

def recommend(movie_title):
    try:
        movie_index = df[df['title']==movie_title].index[0]
        distances = similarities[movie_index]
        movie_objs = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        return [df.iloc[i[0]].title for i in movie_objs]
    except:
        return []
    

print(recommend("Batman"))