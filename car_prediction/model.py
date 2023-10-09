import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

df['name'] =df['name'].str.split(' ').str.slice(0,3).str.join(' ')
df=df[df['year'].str.isnumeric()]
df.loc[:, 'year'] = df['year'].astype('int32')
df.loc[:, 'Price']=df['Price'].str.split(' ').str.get(0).str.replace(',','')
df=df[df['Price'].str.isnumeric()]
df['Price']=df['Price'].astype('int')
df['kms_driven']=df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
df=df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype("int")
df['year_old']=df['year'].apply(lambda x:2021-x)
df.drop("year", axis=1, inplace=True)
df=df.rename(columns={"Price":"price"})

df=df[df['kms_driven']<200000.000000]
df=df[df['price']<4000000]

X = df.drop('price', axis=1)
y = df['price'] 

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

oh_categories = ['name', 'company']
le_categories = ['fuel_type']

ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe.fit(X[oh_categories])

le = LabelEncoder()
X['fuel_type'] = le.fit_transform(X['fuel_type'])

column_trans = ColumnTransformer(
    [('oh', ohe, oh_categories)],
    remainder='passthrough'
)

X=column_trans.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(n_estimators=50,random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=437)

model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMAE: ", mae)
print("RMSE: ", np.sqrt(mse))
print("\nR^2: ", r2)