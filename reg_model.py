# this code is for our regression model to analyse covid data

#Packages for py
import numpy as np
from sklearn.linear_model import LinearRegression

#Data for the model
#this is only sample data

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

#Create the model
model = LinearRegression()

#Model fitting
model.fit(x, y)

#Regression Coefficient and slope etc
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)





