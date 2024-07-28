import numpy as np
import pandas as pd

# Defining the dataset
data = {
    'Day': ['Weekday', 'Weekday', 'Weekday', 'Holiday', 'Saturday', 'Weekday', 'Holiday', 'Sunday', 'Weekday', 'Weekday', 'Saturday', 'Weekday', 'Weekday', 'Weekday', 'Weekday', 'Saturday', 'Weekday', 'Holiday', 'Weekday', 'Weekday',],
    'Season': ['Spring', 'Winter', 'Winter', 'Winter', 'Summer', 'Autumn', 'Summer', 'Summer', 'Winter', 'Summer', 'Spring', 'Summer', 'Winter', 'Summer', 'Winter', 'Autumn', 'Autumn', 'Spring', 'Spring', 'Spring',],
    'Fog': ['None', 'None', 'None', 'High', 'Normal', 'Normal', 'High', 'Normal', 'High', 'None', 'High', 'High', 'Normal', 'High', 'Normal', 'High', 'None', 'Normal', 'Normal', 'Normal',],
    'Rain': ['None', 'Slight', 'None', 'Slight', 'None', 'None', 'Slight', 'None', 'Heavy', 'Slight', 'Heavy', 'Slight', 'None', 'None', 'Heavy', 'Slight', 'Heavy', 'Slight', 'None', 'Heavy',],
    'Class': ['On Time', 'On Time', 'On Time', 'Late', 'On Time', 'Very Late', 'On Time', 'On Time', 'Very Late', 'On Time', 'Cancelled', 'On Time', 'Late', 'On Time', 'Very Late', 'On Time', 'On Time', 'On Time', 'On Time', 'On Time',]
}

# Creating the DataFrame
df = pd.DataFrame(data)
print()

# 5a
P_Class_On = (df['Class'] == 'On Time').sum()/len(df)
P_Class_Late = (df['Class'] == 'Late').sum()/len(df)
P_Class_VeryLate = (df['Class'] == 'Very Late').sum()/len(df)
P_Class_Cancelled = (df['Class'] == 'Cancelled').sum()/len(df)

print(f'P("Class" = "On Time") = {P_Class_On}')
print(f'P("Class" = "Late") = {P_Class_Late}')
print(f'P("Class" = "Very Late") = {P_Class_VeryLate}')
print(f'P("Class" = "Cancelled") = {P_Class_Cancelled}')
print()

# 6c
# X = (Day=Weekday, Season=Winter, Fog=High, Rain=Heavy)

P_x1y = ((df['Day'] == 'Weekday') & (df['Class'] == 'On Time')).sum()/len(df)
P_x1ly = P_x1y/P_Class_On

P_x2y = ((df['Season'] == 'Winter') & (
    df['Class'] == 'On Time')).sum() / len(df)
P_x2ly = P_x2y/P_Class_On

P_x3y = ((df['Fog'] == 'High') & (df['Class'] == 'On Time')).sum() / len(df)
P_x3ly = P_x3y/P_Class_On

P_x4y = ((df['Rain'] == 'Heavy') & (df['Class'] == 'On Time')).sum() / len(df)
P_x4ly = P_x4y/P_Class_On

P_XlY = P_x1ly*P_x2ly*P_x3ly*P_x4ly*P_Class_On
print(f'P(x1|Y) = {P_XlY}')
print()

# 7d
# X = (Day=Weekday, Season=Winter, Fog=High, Rain=Heavy)


P_x1y = ((df['Day'] == 'Weekday') & (df['Class'] == 'Late')).sum()/len(df)
P_x1ly = P_x1y/P_Class_Late

P_x2y = ((df['Season'] == 'Winter') & (df['Class'] == 'Late')).sum() / len(df)
P_x2ly = P_x2y/P_Class_Late

P_x3y = ((df['Fog'] == 'High') & (df['Class'] == 'Late')).sum() / len(df)
P_x3ly = P_x3y/P_Class_Late

P_x4y = ((df['Rain'] == 'Heavy') & (df['Class'] == 'Late')).sum() / len(df)
P_x4ly = P_x4y/P_Class_Late

P_XlY = P_x1ly*P_x2ly*P_x3ly*P_x4ly*P_Class_Late
print(f'P(x1|Y) = {P_XlY}')
print()

# 8a
# X = (Day=Weekday, Season=Winter, Fog=High, Rain=Heavy)


P_x1y = ((df['Day'] == 'Weekday') & (df['Class'] == 'Very Late')).sum()/len(df)
P_x1ly = P_x1y/P_Class_VeryLate

P_x2y = ((df['Season'] == 'Winter') & (
    df['Class'] == 'Very Late')).sum() / len(df)
P_x2ly = P_x2y/P_Class_VeryLate

P_x3y = ((df['Fog'] == 'High') & (df['Class'] == 'Very Late')).sum() / len(df)
P_x3ly = P_x3y/P_Class_VeryLate

P_x4y = ((df['Rain'] == 'Heavy') & (
    df['Class'] == 'Very Late')).sum() / len(df)
P_x4ly = P_x4y/P_Class_VeryLate

P_XlY = P_x1ly*P_x2ly*P_x3ly*P_x4ly*P_Class_VeryLate
print(f'P(x1|Y) = {P_XlY}')
print()

# 9d
# X = (Day=Weekday, Season=Winter, Fog=High, Rain=Heavy)


P_x1y = ((df['Day'] == 'Weekday') & (df['Class'] == 'Cancelled')).sum()/len(df)
P_x1ly = P_x1y/P_Class_Cancelled

P_x2y = ((df['Season'] == 'Winter') & (
    df['Class'] == 'Cancelled')).sum() / len(df)
P_x2ly = P_x2y/P_Class_Cancelled

P_x3y = ((df['Fog'] == 'High') & (df['Class'] == 'Cancelled')).sum() / len(df)
P_x3ly = P_x3y/P_Class_Cancelled

P_x4y = ((df['Rain'] == 'Heavy') & (
    df['Class'] == 'Cancelled')).sum() / len(df)
P_x4ly = P_x4y/P_Class_Cancelled

P_XlY = P_x1ly*P_x2ly*P_x3ly*P_x4ly*P_Class_Cancelled
print(f'P(x1|Y) = {P_XlY}')
print()

# 10a
