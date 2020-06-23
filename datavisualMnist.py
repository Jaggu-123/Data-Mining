from __future__ import print_function
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

train = pd.read_csv("digit-recognizer/train.csv")
test = pd.read_csv("digit-recognizer/test.csv")

y = train['label']
X = train.drop(labels=['label'], axis=1)

print(X.shape, y.shape)

np.random.seed(42)
rndperm = np.random.permutation(train.shape[0])
feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]

plt.gray()
fig = plt.figure(figsize=(16,7))
for i in range(0, 15):
    ax = fig.add_subplot(3, 5, i+1, title="Digit: {}".format(str(train.loc[rndperm[i], 'label'])))
    ax.matshow(train.loc[rndperm[i], feat_cols].values.reshape((28, 28)).astype(float))
# plt.show()

pca = PCA(n_components=3)
pca_results = pca.fit_transform(train[feat_cols].values)

train['pca-one'] = pca_results[:, 0]
train['pca-two'] = pca_results[:, 1]
train['pca-three'] = pca_results[:, 2]

print('Explained {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x = "pca-one", y = "pca-two",
    hue="label",
    # palatte=sns.color_palette("hls", 10),
    data=train.loc[rndperm, :],
    legend="full",
    alpha=0.3
)
plt.show()
