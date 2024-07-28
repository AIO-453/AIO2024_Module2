import numpy as np
import pandas as pd

# Defining the dataset
data = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Overcast', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# Creating the DataFrame
df = pd.DataFrame(data)


def create_train_data(Data_Frame):
    r = Data_Frame.to_numpy()
    r = r[:, 1:]
    return r


train_data = create_train_data(df)
print(train_data)
print()

# 14 a


def compute_prior_probablity(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    # your code here ******************
    prior_probability[0] = (
        (train_data[:, len(train_data[0])-1] == "No").sum())/len(train_data)
    prior_probability[1] = (
        (train_data[:, len(train_data[0])-1] == "Yes").sum())/len(train_data)

    return prior_probability


prior_probablity = compute_prior_probablity(train_data)
print(f"P( play tennis = No) , {prior_probablity[0]}")
print(f"P( play tennis = Yes), {prior_probablity[1]}")
print()
