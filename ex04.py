from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np

diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

model = LinearRegression()

model.fit(X_train, y_train)

coef = model.coef_
intercept = model.intercept_

for i, c in enumerate(X.columns):
    print(f"{c} {coef[i]}")
print("intercept", intercept)

predictions_on_test = model.predict(X_test)
predictions_on_train = model.predict(X_train)

print(predictions_on_test[:10].reshape((-1, 1)))

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print(compute_mse(y_train, predictions_on_train))
print(compute_mse(y_test, predictions_on_test))