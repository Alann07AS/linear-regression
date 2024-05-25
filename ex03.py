from sklearn.model_selection import train_test_split
import numpy as np


X = np.arange(1,21).reshape(10,-1)
y = np.arange(1,11)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Print the results
print("Training Data:")
print("X_train:\n", X_train)
print("y_train:\n", y_train)

print("\nTesting Data:")
print("X_test:\n", X_test)
print("y_test:\n", y_test)