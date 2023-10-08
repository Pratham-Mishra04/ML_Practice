from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

model = load('model.joblib') 
cv = load('vectorizer.joblib')

s = "hi, hello, how are you"

ps = PorterStemmer()

def process_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    final = []
    for t in text:
        if t.isalnum():
            final.append(t)

    text = final[:]
    final.clear()

    for t in text:
        if t not in stopwords.words("english") and t not in string.punctuation:
            final.append(t)

    text = final[:]
    final.clear()

    for t in text:
        final.append(ps.stem(t))

    return " ".join(final)
    

processed_string = process_text(s)

X = cv.transform([processed_string])

prediction = model.predict(X)[0]

if prediction==0:
    print("HAM")
else: 
    print("SPAM")