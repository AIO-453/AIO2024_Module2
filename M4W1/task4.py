
import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data, col_index):
    result = []
    for i in range(len(data)):
        result.append(data[i][col_index])
    return result


def prepare_data(file_name_dataset):
    data = np . genfromtxt(
        file_name_dataset, delimiter=',', skip_header=1) . tolist()

    # get tv ( index =0)
    tv_data = get_column(data, 0)

    # get radio ( index =1)
    radio_data = get_column(data, 1)

    # get newspaper ( index =2)
    newspaper_data = get_column(data, 2)
    # get sales ( index =3)
    sales_data = get_column(data, 3)
    # building X input and y output for training
    # Create list of features for input
    X = [[1, x1, x2, x3]
         for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y


def initialize_params():
    bias = 0
    w1 = random . gauss(mu=0.0, sigma=0.01)
    w2 = random . gauss(mu=0.0, sigma=0.01)
    w3 = random . gauss(mu=0.0, sigma=0.01)
    # comment this line for real application
    return [0, -0.01268850433497871, 0.004752496982185252, 0.0073796171538643845]
    # return [bias , w1 , w2 , w3]

# Predict output by using y = x0*b + x1*w1 + x2*w2 + x3*w3


def predict(X_features, weights):
    # your code here ......
    result = 0
    for i in range(len(X_features)):
        result += X_features[i] * weights[i]
    return result


def compute_loss(y_hat, y):
    return (y_hat - y) ** 2

# compute gradient


def compute_gradient_w(X_features, y, y_hat):
    # your code here ......
    dl_dweights = []
    for i in range(len(X_features)):
        dl_dw = 2 * (y_hat - y) * X_features[i]
        dl_dweights.append(dl_dw)
    return dl_dweights

    # update weights


def update_weight(weights, dl_dweights, lr):
    # your code here ......
    for i in range(len(weights)):
        weights[i] = weights[i] - lr * dl_dweights[i]
    return weights


def implement_linear_regression(X_feature, y_ouput, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()
    N = len(y_ouput)
    for epoch in range(epoch_max):
        # print (" epoch ", epoch )
        for i in range(N):
            # get a sample - row i
            features_i = X_feature[i]

            y = y_ouput[i]
            # compute output
            y_hat = predict(features_i, weights)
            # compute loss
            loss = compute_loss(y, y_hat)
            # compute gradient w1 , w2 , w3 , b
            dl_dweights = compute_gradient_w(features_i, y, y_hat)
            # update parameters
            weights = update_weight(weights, dl_dweights, lr)
            # logging
            losses . append(loss)
    return weights, losses


if __name__ == '__main__':

    data_path = './advertising.csv'
    X, y = prepare_data(data_path)
    W, L = implement_linear_regression(X, y)
    plt . plot(L[0:100])
    plt . xlabel("# iteration ")
    plt . ylabel(" Loss ")
    plt . show()
