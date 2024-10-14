from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae

df = pd.read_csv('./BTC-Daily.csv')
df = df.drop_duplicates()

# Range of dates covered
df['date'] = pd.to_datetime(df['date'])
date_range = str(df['date'].dt.date.min()) + ' to ' + \
    str(df['date'].dt.date.max())
print(date_range)

# Tạo cột năm, tháng, ngày từ cột 'date'
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Giả sử unique_years chứa danh sách các năm có trong dữ liệu
unique_years = df['year'].unique()

# Filter data for 2019-2022
df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31')]

# Convert 'date' column to datetime objects before applying date2num
df_filtered['date'] = df_filtered['date'].map(mdates.date2num)


# Create candlestick chart
fig, ax = plt.subplots(figsize=(20, 6))

# Correctly pass data to candlestick_ohlc using .values and specifying all required columns
candlestick_ohlc(ax, df_filtered[['date', 'open', 'high', 'low', 'close']].values,
                 width=0.6, colorup='g', colordown='r')

# Format x-axis as dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

# Add title and labels
plt.title('Bitcoin Candlestick Chart (2019-2022)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()


def predict(x, w, b):
    # ##### Your code here ##### 5d
    return x.dot(w) + b


def gradient(y_hat, y, x):
    # ##### Your code here ##### 6b
    loss = y_hat - y
    dw = x . T . dot(loss) / len(y)
    db = np . sum(loss) / len(y)
    cost = np . sum(loss**2) / (2 * len(y))
    return (dw, db, cost)


def update_weight(w, b, lr, dw, db):
    # ##### Your code here #####  7d
    w_new = w - dw * lr
    b_new = b - db * lr
    return (w_new, b_new)

# 8c


def linear_regression_vectorized(X, y, learning_rate=0.01, num_iterations=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias
    losses = []
    # ##### Your code here #####
    for _ in range(num_iterations):
        y_hat = predict(X, w, b)
        dw, db, cost = gradient(y_hat, y, X)
        w, b = update_weight(w, b, learning_rate, dw, db)
        losses . append(cost)

    return w, b, losses


scalar = StandardScaler()

df["Standardized_Close_Prices"] = scalar.fit_transform(
    df["close"].values.reshape(-1, 1))
df["Standardized_Open_Prices"] = scalar.fit_transform(
    df["open"].values.reshape(-1, 1))
df["Standardized_High_Prices"] = scalar.fit_transform(
    df["high"].values.reshape(-1, 1))
df["Standardized_Low_Prices"] = scalar.fit_transform(
    df["low"].values.reshape(-1, 1))


# Converting Date to numerical form

df['date_str'] = df['date'].dt.strftime('%Y%m%d%H%M%S')

# Convert the string date to a numerical value
df['NumericalDate'] = pd.to_numeric(df['date_str'])

# Drop the intermediate 'date_str' column if not needed
df.drop(columns=['date_str'], inplace=True)


X = df[["NumericalDate", "Standardized_Open_Prices",
        "Standardized_High_Prices", "Standardized_Low_Prices"]]
y = df["Standardized_Close_Prices"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=True)

# lr = LinearRegression()
# lr.fit(X_train, y_train)

X = df[["Standardized_Open_Prices",
        "Standardized_High_Prices", "Standardized_Low_Prices"]]
y = df["Standardized_Close_Prices"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True)

b = 0
w = np.zeros(X_train.shape[1])
lr = 0.01
epochs = 200


X = df[["Standardized_Open_Prices",
        "Standardized_High_Prices", "Standardized_Low_Prices"]]
y = df["Standardized_Close_Prices"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True)

b = 0
w = np.zeros(X_train.shape[1])
lr = 0.01
epochs = 200

w, b, losses = linear_regression_vectorized(
    X_train.values, y_train.values, lr, epochs)

# Plot the loss function
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function during Gradient Descent')
plt.show()


# Make predictions on the test set
y_pred = predict(X_test, w, b)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate MAE
mae = np.mean(np.abs(y_pred - y_test))

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# Calculate R-squared on training data
y_train_pred = predict(X_train, w, b)
train_accuracy = r2_score(y_train, y_train_pred)

# Calculate R-squared on testing data
test_accuracy = r2_score(y_test, y_pred)

print("Root Mean Square Error (RMSE):", round(rmse, 4))
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Training Accuracy (R-squared):", round(train_accuracy, 4))
print("Testing Accuracy (R-squared):", round(test_accuracy, 4))

# Filter data for 2019-01-01 to 2019-04-01
df_2019_q1 = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2019-04-01')]

# Prepare X and y for prediction
X_2019_q1 = df_2019_q1[["open", "high", "low"]]
y_2019_q1_actual = df_2019_q1["close"]

# Giả sử hàm dự đoán là 'predict' và các tham số mô hình là 'w' và 'b'
y_2019_q1_pred = predict(X_2019_q1, w, b)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df_2019_q1['date'], y_2019_q1_actual,
         label='Actual Close Price', marker='o')
plt.plot(df_2019_q1['date'], y_2019_q1_pred,
         label='Predicted Close Price', marker='x')
plt.title('Actual vs. Predicted Bitcoin Close Price (01/01/2019 - 04/01/2019)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
