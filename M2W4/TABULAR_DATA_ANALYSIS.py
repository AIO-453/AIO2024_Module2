import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def pearson_corr(X, Y):
    r = np.sum((X - X.mean()) * (Y - Y.mean())) / \
        np.sqrt(np.sum((X - X.mean())**2) * np.sum((Y - Y.mean())**2))
    return r


data = pd.read_csv('./advertising.csv')
x = data['TV']
y = data['Radio']

corr_xy = pearson_corr(x, y)
print(f"5b Correlation between TV and Sales : {round(corr_xy, 2)}")


def correlation(x, y):
    # Your code here #
    r = np.sum((X - X.mean()) * (Y - Y.mean())) / \
        np.sqrt(np.sum((X - X.mean())**2) * np.sum((Y - Y.mean())**2))
    return r


def pearson_corr(X, Y):
    r = np.sum((X - X.mean()) * (Y - Y.mean())) / \
        np.sqrt(np.sum((X - X.mean())**2) * np.sum((Y - Y.mean())**2))
    return r


features = ['TV', 'Radio', 'Newspaper']

for feature_1 in features:
    for feature_2 in features:
        correlation_value = pearson_corr(data[feature_1], data[feature_2])
        print(f" Correlation between {feature_1} and {
              feature_2}: {round(correlation_value, 2)}")

# Question 13


plt . figure(figsize=(10, 8))
# Your code here #
data_corr = data.corr()
sns.heatmap(data_corr, annot=True, fmt=".2f", linewidth=.5)
plt . show()
