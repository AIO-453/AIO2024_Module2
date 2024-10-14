# import libraries
import numpy as np
import matplotlib.pyplot as plt
import random


def mean_normalization(X):
    N = len(X)
    maxi = np.max(X)
    mini = np.min(X)
    avg = np.mean(X)
    X = (X-avg) / (maxi-mini)
    X_b = np.c_[np.ones((N, 1)), X]
    return X_b, maxi, mini, avg


def batch_gradient_descent(X_b, y, n_epochs=100, learning_rate=0.01):
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033],
                         [0.29763545]])
    thetas_path = [thetas]
    losses = []
    N = X_b.shape[0]

    for i in range(n_epochs):
        y_pred = X_b.dot(thetas)
        residuals = (y_pred-y)

        loss = (residuals) ** 2
        gradients = 2*(X_b.T.dot(residuals))/(N)

        thetas = thetas - learning_rate * gradients
        thetas_path.append(thetas)
        loss_mean = np.sum(loss) / N
        losses.append(loss_mean)
    return thetas_path, losses


# dataset
data = np.genfromtxt('./advertising.csv', delimiter=',', skip_header=1)
N = data.shape[0]
X = data[:, :3]
y = data[:, 3:]

# Normalize input data by using mean normalizaton
X_b, maxi, mini, avg = mean_normalization(X)

bgd_thetas, losses = batch_gradient_descent(
    X_b, y, n_epochs=100, learning_rate=0.01)
print(f'3c: {round(sum(losses), 2)}')

bgd_thetas, losses = batch_gradient_descent(
    X_b, y, n_epochs=100, learning_rate=0.01)

x_axis = list(range(100))

plt . plot(x_axis, losses[:100], color="r")
plt . show()
