from sklearn . preprocessing import StandardScaler
from sklearn . model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./SalesPrediction.csv')
df = pd.get_dummies(df)
df = df.fillna(df.mean())
# df.isnull().sum()
# df.corr()
# new_df = df[['TV', 'Radio', 'Social Media', 'Sales']]
# sns.heatmap(new_df.corr(numeric_only=True), cmap='YlGnBu', annot=True)
# plt.show()
# sns.pairplot(data=df, x_vars=['TV', 'Radio', 'Social Media'],
#              y_vars='Sales',
#              height=5,
#              kind='reg')

# plt.show()

# sns.pairplot(data=df, x_vars=['Influencer_Macro', 'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano'],
#              y_vars='Sales',
#              height=5,
#              kind='reg')

# plt.show()

X = df[['TV', 'Radio', 'Social Media', 'Sales', 'Influencer_Macro',
       'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]

X.head(5)
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

preds = regressor.predict(X_test)
print(r2_score(y_test, preds))

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


regressor = LinearRegression()
regressor.fit(X_poly, y_train)

preds = regressor.predict(X_test_poly)
print(r2_score(y_test, preds))


# Handle Null values
df = df . fillna(df . mean())
# Get features
X = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro',
        'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]
y = df[['Sales']]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=0
)

scaler = StandardScaler()
X_train_processed = scaler . fit_transform(X_train)
print(scaler . mean_[0])
