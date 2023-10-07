import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

housing = pd.read_csv("housing.csv")

housing["LSTATRM"] = housing['LSTAT']/housing['RM']

# Since were are splitting data on basis on CHAS as the data is very imbalance about it, CHAS cannot contian any null values
chas_median = housing['CHAS'].median()
housing['CHAS'].fillna(chas_median, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

x_train = strat_train_set.drop('MEDV', axis=1)
y_train = strat_train_set['MEDV']
x_test = strat_test_set.drop('MEDV', axis=1)
y_test = strat_test_set['MEDV']

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("standardizer", StandardScaler())
])

x_train_prepared = pipeline.fit_transform(x_train) 

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [10, 50, 100, 200, 300],
}

grid=GridSearchCV(RandomForestRegressor(), param_grid, verbose=3)
grid.fit(x_train_prepared, y_train)

model = grid.best_estimator_

scores = cross_val_score(model, x_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print("Scores:", rmse_scores)
print("Mean: ", rmse_scores.mean())
print("Standard deviation: ", rmse_scores.std())

x_test_prepared = pipeline.transform(x_test)
predictions = model.predict(x_test_prepared)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = explained_variance_score(y_test, predictions)

print("\nMAE: ", mse)
print("MSE: ", mse)
print("RMSE: ", np.sqrt(mse))
print("\nR^2: ", r2)

dump(model, 'house_prediction.joblib') 