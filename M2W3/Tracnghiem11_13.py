import numpy as np
import pandas as pd

# Defining the dataset
data = {
    'Length': [1.4, 1.0, 1.3, 1.9, 2.0, 1.8, 3.0, 3.8, 4.1, 3.9, 4.2, 3.4,],
    'Class': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Creating the DataFrame
df = pd.DataFrame(data)
print()

# 11a
mean_0 = df['Length'][df['Class'] == 0].to_numpy().mean()
var_0 = df['Length'][df['Class'] == 0].to_numpy().var()
std_0 = df['Length'][df['Class'] == 0].to_numpy().std()

print(f'mean_0 = {mean_0}, var_0 = {var_0}')
print()

# 12b
mean_1 = df['Length'][df['Class'] == 1].to_numpy().mean()
var_1 = df['Length'][df['Class'] == 1].to_numpy().var()
std_1 = df['Length'][df['Class'] == 1].to_numpy().std()


print(f'mean_1 = {mean_1}, var_1 = {var_1}')
print()

# 13


def gaussian_naive_baiyes(x, std, mean):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))


P_class_0 = 0.5
P_class_1 = 0.5

# P(len=3.4) = P(len=3.4|class=0)*P(class=0) + P(len=3.4|class=1)*P(class=1)
P_len_34 = gaussian_naive_baiyes(3.4, np.sqrt(
    var_0), mean_0)*P_class_0 + gaussian_naive_baiyes(3.4, np.sqrt(var_1), mean_1)*P_class_1
# print(P_len_34)

# P(class=0|len=3.4) = P(len=3.4|class=0)*P(class=0)/P(len=3.4)
P_class0_34 = gaussian_naive_baiyes(
    3.4, np.sqrt(var_0), mean_0)*P_class_0/P_len_34
print(P_class0_34)

# P(class=1|len=3.4) = P(len=3.4|class=1)*P(class=1)/P(len=3.4)
P_class1_34 = gaussian_naive_baiyes(
    3.4, np.sqrt(var_1), mean_1)*P_class_1/P_len_34
print(P_class1_34)
