# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


dataset_path = '/Problem3.csv'
data_df = pd.read_csv(dataset_path)

categorical_cols = data_df.select_dtypes(include=['object', 'bool']).columns.to_list()
ordinal_encoder = OrdinalEncoder()
encode_categorical_cols= ordinal_encoder.fit_transform(data_df[categorical_cols])

encode_categorical_df = pd.DataFrame(encode_categorical_cols, columns=categorical_cols)
numerical_df = data_df.drop(columns=categorical_cols, axis=1)
encode_df = pd.concat([encode_categorical_df, numerical_df], axis=1)

X = encode_df.drop(columns=['area'], axis=1)
y = encode_df['area']

test_size = 0.3
random_state = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

xg_reg = xgb.XGBRegressor(seed=7, learning_rate=0.1, n_estimators=102, max_depth=3)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print ('Evaluation results on test set:')
print (f'Mean Absolute Error: {mae}')
print (f'Mean Squared Error : {mse}')