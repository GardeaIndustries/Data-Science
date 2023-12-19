# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Identify missing data (assuming that missing data is represented as NaN)
missing_data = dataset.isnull().sum()

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling is done in order to avoid some features to be dominated by other features in such a way that the dominated features are ignored by the model
sc = StandardScaler()
X_train[:, 3: ] = sc.fit_transform(X_train[:, 3: ])
X_test[:, 3: ] = sc.transform(X_test[:, 3: ])

