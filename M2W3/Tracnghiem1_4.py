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

# Printing the DataFrame
print(df)
print()

# 1a
yes_count = df['PlayTennis'].value_counts()['Yes']
no_count = df['PlayTennis'].value_counts()['No']

P_playtennis = len(df)
P_playtennis_Y = yes_count / P_playtennis
P_playtennis_N = no_count / P_playtennis
print(f'P("Play Tennis" = "Yes"= {
      P_playtennis_Y}), P("Play Tennis" = "No")= {P_playtennis_N}')
print()

# 2b
# X = (Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong)
# P_(x1|Y) = P(x1 n Y)/P(Y)

P_x1y = ((df['Outlook'] == 'Sunny') & (
    df['PlayTennis'] == 'Yes')).sum() / len(df)
P_x1ly = P_x1y/P_playtennis_Y

P_x2y = ((df['Temperature'] == 'Cool') & (
    df['PlayTennis'] == 'Yes')).sum() / len(df)
P_x2ly = P_x2y/P_playtennis_Y

P_x3y = ((df['Humidity'] == 'High') & (
    df['PlayTennis'] == 'Yes')).sum() / len(df)
P_x3ly = P_x3y/P_playtennis_Y

P_x4y = ((df['Wind'] == 'Strong') & (
    df['PlayTennis'] == 'Yes')).sum() / len(df)
P_x4ly = P_x4y/P_playtennis_Y

P_XlY = P_x1ly*P_x2ly*P_x3ly*P_x4ly*P_playtennis_Y
print(f'P(x1|Y) = {P_XlY}')
print()

# 3c
P_x1n = ((df['Outlook'] == 'Sunny') & (
    df['PlayTennis'] == 'No')).sum() / len(df)
P_x1ln = P_x1n/P_playtennis_N

P_x2n = ((df['Temperature'] == 'Cool') & (
    df['PlayTennis'] == 'No')).sum() / len(df)
P_x2ln = P_x2n/P_playtennis_N

P_x3n = ((df['Humidity'] == 'High') & (
    df['PlayTennis'] == 'No')).sum() / len(df)
P_x3ln = P_x3n/P_playtennis_N

P_x4n = ((df['Wind'] == 'Strong') & (df['PlayTennis'] == 'No')).sum() / len(df)
P_x4ln = P_x4n/P_playtennis_N

P_XlN = P_x1ln*P_x2ln*P_x3ln*P_x4ln*P_playtennis_N
print(f'P(x1|N) = {P_XlN}')
print()

# 4a
print(f'P(x1|Y) = {P_XlY} > P(x1|N) = {P_XlN}')
