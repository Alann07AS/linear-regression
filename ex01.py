from sklearn.linear_model import LinearRegression
import numpy as np

X, y = [[1],[2.1],[3]], [[1],[2],[3]]
# Data
X = np.array([[1], [2.1], [3]])
y = np.array([1, 2, 3])

# Linear Regression model
model = LinearRegression()

# Fitting the model
model.fit(X, y)

# Predicting for x_pred = [[4]]
x_pred = np.array([[4]])
y_pred = model.predict(x_pred)

# Print coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Print the score of the regression (R^2)
score = model.score(X, y)
print("Regression Score (R^2):", score)
