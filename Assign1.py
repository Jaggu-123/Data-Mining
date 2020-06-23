import numpy as np
import pandas as pd

df = pd.read_csv('weather.csv')

# Find the null values
print(df.isnull().sum())

# converting the categorical features
df['RainToday'] = pd.get_dummies(df['RainToday'])
df['RainTomorrow'] = pd.get_dummies(df['RainTomorrow'])

# Filling in the missing values
df.fillna(df.mean(), inplace=True)

dfLabels = df['RainTomorrow']
df = df.drop(['RainTomorrow'], axis=1)

# Splitting the dataframe
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, dfLabels, test_size=0.33, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

data = pd.concat([X_train, X_test])
print(data.shape)
print(df.shape)