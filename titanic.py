import numpy as np
import pandas as pd
import math

datasets = pd.read_csv('titanic/train.csv')
trainLabels = datasets['Survived']
# train = datasets.drop(['Survived'], axis=1)
train = datasets.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
train = train.values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
train[:, 2:3] = imputer.fit_transform(train[:, 2:3])
train[:, -1] = ['C' if type(x) == float and math.isnan(x) else x for x in train[:, 7]]
for i in range(0, train.shape[0]):
    if type(train[i][6]) == float and math.isnan(train[i][6]):
        train[i][6] = 'L'
    else:
        train[i][6] = train[i][6][0]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
train[:, 1] = labelencoder.fit_transform(train[:, 1])
labelencoder_1 = LabelEncoder()
train[:, 7] = labelencoder_1.fit_transform(train[:, 7])
labelencoder_2 = LabelEncoder()
train[:, 6] = labelencoder_2.fit_transform(train[:, 6])
train = pd.DataFrame(train)
data = pd.get_dummies(train[7])
data_1 = pd.get_dummies(train[6])
data = data.drop([0], axis=1)
data_1 = data_1.drop([0], axis=1)
train = pd.concat([train, data, data_1], axis=1)
train = train.drop([6, 7], axis=1)
train = train.values

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train = scaler.fit_transform(train)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(train)
# X_test = pca.transform(test)
explained_ratio = pca.explained_variance_ratio_
print(explained_ratio)
X_train['pca-one'] = X_train[:, 0]
X_train['pca-two'] = X_train[:, 1]

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x = "pca-one", y = "pca-two",
    hue="Survived",
    # palatte=sns.color_palette("hls", 10),
    data=X_train.loc[:, :],
    legend="full",
    alpha=0.3
)
plt.show()

print(train)
print(train.shape)
