 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)
# print(X,"\n","\n",y)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# print(X,"\n","\n",y)

# Training the SVR model on the whole data set
regressor = SVR(kernel= 'rbf').fit(X,y)

# Predict a new result
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)))

# Visualize the Results
# plt.scatter( sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red' )
# plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color='blue')
# plt.title('Compare Salary vs Position Level (SVR)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Compare Salary vs Level (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
