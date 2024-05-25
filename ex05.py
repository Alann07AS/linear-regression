from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

X, y, coef = make_regression(n_samples=100,
                        n_features=1,
                        n_informative=1,
                        noise=10,
                        coef=True,
                        random_state=0,
                        bias=100.0)
# print(X)
X = X.reshape(1,-1)[0] #to have same shape as y
# X = X.transpose()[0] #to have same shape as y


plt.scatter(X, y)
# plt.show()


def compute_mse(coef_intercept, X, y):
    '''
    coef_intercept is a list that contains a and b: [a,b]
    X is the features set
    y is the target

    Returns a float which is the MSE
    '''
    coef, intercept = coef_intercept

    y_preds = coef * X + intercept
    mse = ((y_preds-y)**2).mean() #mean_squared_error(y, y_preds)

    return mse


print(compute_mse([1, 2], X, y))

import numpy as np

aa, bb = np.mgrid[-200:200:0.5, -200:200:0.5]
grid = np.c_[aa.ravel(), bb.ravel()]

print(len(grid))
print(grid)

from multiprocessing import Pool
from functools import partial

with Pool() as pool:
        partial_compute_mse = partial(compute_mse, X=X, y=y)
        # Use pool.map to parallelize the computation
        losses = pool.map(partial_compute_mse, grid)
# losses = [compute_mse(a_b, X, y) for a_b in grid]
losses = np.array(losses)
losses_reshaped = losses.reshape(aa.shape)

f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(aa,
                    bb,
                    losses_reshaped,
                    100,
                    cmap="RdBu",
                    vmin=0,
                    vmax=160000)
ax_c = f.colorbar(contour)
ax_c.set_label("MSE")

ax.set(aspect="equal",
    xlim=(-200, 200),
    ylim=(-200, 200),
    xlabel="$a$",
    ylabel="$b$")

x_i, y_i =  np.unravel_index(losses_reshaped.argmin(), losses_reshaped.shape)
ax.scatter(aa[x_i, y_i], bb[x_i, y_i], s=12, color="green")

plt.show()