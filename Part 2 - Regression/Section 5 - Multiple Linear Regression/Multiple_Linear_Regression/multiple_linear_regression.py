# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

## Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
## Feature Scaling
#"""from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""
#
## Fitting Multiple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = regressor.predict(X_test)

# Baxkward Elimintation 2 1 4 5
# Adj R**2: .939 -> .944 -> .946 -> 0.947 -> 944
import statsmodels.api as sm
X = np.append(np.ones((50,1)).astype(int), X, axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_otp_train = X_train[:,[0,3,5]]
X_otp_test = X_test[:,[0,3,5]]

regressor_OLS = sm.OLS(y_train, X_otp_train).fit()
summary = regressor_OLS.summary()
y_opt_pred = regressor_OLS.predict(X_otp_test)
print(summary)



