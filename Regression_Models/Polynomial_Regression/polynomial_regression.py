# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Training the Linear Regression model on the whole dataset
lin_reg = LinearRegression().fit(X, y)


# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression().fit(X_poly, y)

# Visualize the linear regression results (Linear regression is not well adapted to the results, this results in an ineffective model.)
# plt.scatter(X, y, color='red' )
# plt.plot(X, lin_reg.predict(X), color='blue')
# plt.title('Compare Salary vs Level (Linear Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

# Visualize polynomal regression
# plt.scatter(X, y, color='red' )
# plt.plot(X,lin_reg_2.predict((poly_reg.fit_transform(X)), color='blue'))
# plt.title('Compare Salary vs Level (Polynomial Regression)')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Compare Salary vs Level (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression (unreliable)
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression 
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))