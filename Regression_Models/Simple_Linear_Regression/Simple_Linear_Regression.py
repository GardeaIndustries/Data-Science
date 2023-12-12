# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = X = dataset.iloc[:, :-1].values  # dataset..drop('Salary', axis=1) is also an alternative
y = dataset.iloc[:, -1].values # y = dataset['Salary'] is also an alternative


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the model on the trainig set
regressor = LinearRegression().fit(X_train, y_train)

# Predicting the Test set results
y_pred_test = regressor.predict(X_test)
# Predicting the Training set results
y_pred_train = regressor.predict(X_train)


# Visualizing the Training set Results
# 2D plot with X axis being years of experience, and y axis being the salary range
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, y_pred_train, color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.show()

# Visualize the Test Set Results 
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test, y_pred_test, color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
