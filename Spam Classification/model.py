import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
# import ssl
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("spam.csv")

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={"v1":"target","v2":"text"}, inplace=True)

le = LabelEncoder()
df['target']=le.fit_transform(df['target'])

df.drop_duplicates(keep="first", inplace=True)

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download("punkt")
# nltk.download('stopwords')

ps = PorterStemmer()

def process_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    final = []
    for t in text:
        if t.isalnum():
            final.append(t)

    text = final[:] #Copy by value
    final.clear()

    for t in text:
        if t not in stopwords.words("english") and t not in string.punctuation:
            final.append(t)

    text = final[:]
    final.clear()

    for t in text:
        final.append(ps.stem(t))

    return " ".join(final)

df['processed_text'] = df['text'].apply(process_text)

cv = CountVectorizer()

X = cv.fit_transform(df['processed_text']).toarray()
y = df['target']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(X, y):
    strat_train_set = X[train_index], y.iloc[train_index]
    strat_test_set = X[test_index], y.iloc[test_index]

X_train, y_train = strat_train_set
X_test, y_test = strat_test_set

etc = ExtraTreesClassifier(n_estimators=200, max_depth=50, min_samples_split=10, random_state=101)

etc.fit(X_train, y_train)

predictions = etc.predict(X_test)

print("Classification Report:\n", classification_report(y_test, predictions))

conf_matrix = confusion_matrix(y_test, predictions)
conf_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

print("Confusion Matrix:\n", conf_df)