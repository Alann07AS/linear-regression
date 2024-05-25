from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# import random 
# r = random.randint(0,100)
# print(r)
r=0
X, y, coef = make_regression(n_samples=100,
                         n_features=1,
                         n_informative=1,
                         noise=10,
                         coef=True,
                         random_state=r,
                         bias=100.0)
# print(coef)

plt.scatter(X, y, label='Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("X and y")


# Fit Linear Regression
model = LinearRegression()
model.fit(X, y)

# Get coefficients and intercept
coef = model.coef_[0]
intercept = model.intercept_

print(coef)
print(intercept)

x_range = np.linspace(min(X), max(X), 100)
y_fit = coef * x_range + intercept

# print(x_range)
# print(y_fit)

plt.plot(x_range, y_fit, color='red', linewidth=2)
# label=f'Fitted Line: y = {coef:.2f} * x + {intercept:.2f}'

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

y_pred = model.predict(X)
print("Mean Squared Error (MSE):", compute_mse(y, y_pred))

plt.show()