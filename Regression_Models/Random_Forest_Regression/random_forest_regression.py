# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the model on the whole dataset
regressor =RandomForestRegressor(max_depth=2, n_estimators=10, random_state=0).fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualizd the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Compare Salary vs Level (Random Forest Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()