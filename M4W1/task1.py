import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data, col_index):
    result = []
    for i in range(len(data)):
        result.append(data[i][col_index])
    return result


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()
    N = len(data)

    # get tv (index=0)
    tv_data = get_column(data, 0)

    # get radio (index=1)
    radio_data = get_column(data, 1)

    # get newspaper (index=2)
    newspaper_data = get_column(data, 2)

    # get sales (index=3)
    sales_data = get_column(data, 3)

    # build X input and y output for training
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data

    return X, y


def initialize_params():
    w1, w2, w3, b = (0.016992259082509283,
                     0.0070783670518262355, -0.002307860847821344, 0)
    return w1, w2, w3, b


def predict(x1, x2, x3, w1, w2, w3, b):
    return w1 * x1 + w2 * x2 + w3 * x3 + b


def compute_loss_mse(y_hat, y):
    return (y_hat - y) ** 2


def compute_gradient_wi(x, y_hat, y):
    return 2 * (y_hat - y) * x


def compute_gradient_b(y_hat, y):
    return 2 * (y_hat - y)


def update_weight_wi(wi, dl_dwi, lr):
    return wi - lr * dl_dwi


def update_weight_b(b, dl_db, lr):
    return b - lr * dl_db


def implement_linear_regression(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []
    w1, w2, w3, b = initialize_params()
    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
            # get a sample
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]

            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # compute loss
            loss = compute_loss_mse(y_hat, y)

            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y_hat, y)
            dl_dw2 = compute_gradient_wi(x2, y_hat, y)
            dl_dw3 = compute_gradient_wi(x3, y_hat, y)
            dl_db = compute_gradient_b(y_hat, y)

            #  update parameters
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            losses.append(loss)
    return (w1, w2, w3, b, losses)


if __name__ == '__main__':

    data_path = './advertising.csv'
    X, y = prepare_data(data_path)

    lst = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
    print(f'Question 1a: {lst}')

    y = predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
    print(f'Question 2a: {y}')

    l = compute_loss_mse(y_hat=1, y=0.5)
    print(f'Question 3a: {l}')

    g_wi = compute_gradient_wi(x=1.0, y=1.0, y_hat=0.5)
    print(f'Question 4a: {g_wi}')

    g_b = compute_gradient_b(y=2.0, y_hat=0.5)
    print(f'Question 5b: {g_b}')

    after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
    print(f'Question 6a: {after_wi}')

    after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
    print(f'Question 7a: {after_b}')

    data_path = './advertising.csv'
    X, y = prepare_data(data_path)
    (w1, w2, w3, b, losses) = implement_linear_regression(X, y)
    plt . plot(losses[:100])
    plt . xlabel("# iteration ")
    plt . ylabel(" Loss ")
    plt . show()
    print(f'Question 8a: {w1, w2, w3}')

    # given new data
    tv = 19.2
    radio = 35.9
    newspaper = 51.3
    sales = predict(tv, radio, newspaper, w1, w2, w3, b)
    print(f'Question 9b: predicted sales is {sales}')

    def compute_loss_mae(y_hat, y):
        return abs(y_hat - y)

    l = compute_loss_mae(y_hat=1, y=0.5)
    print(f'Question 10a: {l}')


# 10a
