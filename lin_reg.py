import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("housing.csv")

# removing null values
data.dropna(inplace=True)

# assigning data and target
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# feature egineering
data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

data['household_rooms'] = data['total_rooms'] / data['households']
data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']

# print(data.info())

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, y_train = data.drop(['median_house_value'], axis=1), data['median_house_value']
X_test, y_test = data.drop(['median_house_value'], axis=1), data['median_house_value']


def heatmapp():
    plt.figure(figsize=(15, 9))
    sns.heatmap(data.corr(), annot=True, cmap='YlGnBu')
    plt.show()


# model


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)


def linear_model():
    lin_model = LinearRegression()
    lin_model.fit(X_train_s, y_train)
    scorlin = lin_model.score(X_test,y_test)
    print(scorlin)


# using Random Regressor

def ranfor_model():
    forest = RandomForestRegressor()

    forest.fit(X_train, y_train)
    scor = forest.score(X_test, y_test)
    print(scor)


# using cross validation

# param_grid = {
#     'n_estimators' : [100, 200, 300],
#     'min_samples_split' : [2, 4, 6, 8],
#     'max_depth' : [None, 4, 6, 8]
# }
# grid_search = GridSearchCV(forest,param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True)
# grid_search.fit(X_train,y_train)

# best_forest = grid_search.best_estimator_
# print(best_forest)
# print(best_forest.score(X_test,y_test))

#
# predcition = forest.predict(user_input)

# print("actual: ", y_test[69])
# print("PRED: ", predcition)
