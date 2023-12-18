# While the Decision Tree Regression Model may not be ideally suited for two-dimensional datasets, this implementation can be readily adapted to any other dataset.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0).fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Viualizing the Decision Tree Regression results(higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Compare Salary vs Level (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
