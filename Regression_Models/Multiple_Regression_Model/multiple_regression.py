# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data using one hot encoding
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(),[3] )], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train the model on the Training set
regressor = LinearRegression().fit(X_train, y_train)

# Predicting the Test set results
y_pred_test = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_test.reshape(len(y_pred_test), 1), y_test.reshape(len(y_test), 1)), axis= 1 , out=None))


# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
