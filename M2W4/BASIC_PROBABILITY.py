import numpy as np


def compute_mean(X):
    return sum(X)/len(X)


X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]

print(f'1a compute mean: {compute_mean(X)}')


# Question 2
def compute_median(X):
    size = len(X)
    X = np.sort(X)
    if size % 2 == 0:
        return (X[size // 2 - 1] + X[size // 2]) / 2
    else:
        return X[size // 2]


X = [1, 5, 4, 4, 9, 13]
print("2b Median : ", compute_median(X))


# Question 3
def compute_std(X):
    mean = compute_mean(X)
    variance = 0
    X = np.array(X)
    # your code here *******************
    variance = X.var()
    return np . sqrt(variance)


X = [171, 176, 155, 167, 169, 182]
print(f'3c compute std{compute_std(X)}')

# Question 4


def compute_correlation_cofficient(X, Y):
    N = len(X)
    numerator = np.sum((X - X.mean()) * (Y - Y.mean()))
    denominator = np.sqrt(np.sum((X - X.mean())**2)
                          * np.sum((Y - Y.mean())**2))
    # your code here ****************

    return np . round(numerator / denominator, 2)


X = np . asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np . asarray([4, 25, 121, 36, 16, 225, 81])
print("4d Correlation : ", compute_correlation_cofficient(X, Y))
