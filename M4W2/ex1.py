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


def stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.00001):
    # thetas = np.random.randn(X_b.shape[1],1)
    thetas = np.asarray([[1.16270837], [-0.81960489],
                        [1.39501033], [0.29763545]])
    thetas_path = [thetas]
    losses = []  # Initialize as an empty list

    N = X_b.shape[0]

    for epoch in range(n_epochs):
        for i in range(N):
            # random_index = random.randint(N)
            random_index = i

            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]

            # Compute output
            y_pred = xi.dot(thetas)

            # Compute loss li
            loss = (y_pred - yi) ** 2 / 2

            # Compute gradient for loss
            gradients_loss = y_pred - yi

            # Compute gradient
            gradients = xi.T.dot(gradients_loss)

            # Update theta
            thetas = thetas - learning_rate * gradients
            thetas_path.append(thetas)

            # Append only the scalar loss value to the losses list
            losses.append(loss[0][0])  # or losses.append(loss[0][0])

    return thetas_path, losses


# dataset
data = np.genfromtxt('./advertising.csv', delimiter=',', skip_header=1)
N = data.shape[0]
X = data[:, :3]
y = data[:, 3:]

# Normalize input data by using mean normalizaton
X_b, maxi, mini, avg = mean_normalization(X)

sgd_theta, losses = stochastic_gradient_descent(
    X_b, y, n_epochs=1, learning_rate=0.01)
print(f'1b: {np.sum(losses)}')

sgd_theta, losses = stochastic_gradient_descent(
    X_b, y, n_epochs=50, learning_rate=0.01)


x_axis = list(range(500))
plt.plot(x_axis, losses[:500], color="r")
plt.show()
