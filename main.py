import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("data-science-london-scikit-learn/train.csv", header=None)
trainLabels = pd.read_csv("data-science-london-scikit-learn/trainLabels.csv", header=None)
test = pd.read_csv("data-science-london-scikit-learn/test.csv", header=None)

# print(plt.style.available)
plt.style.use('ggplot')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split

X, y = train, np.ravel(trainLabels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Scaling the dataset
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
scaler1 = Normalizer()
X_sc = scaler1.fit_transform(X)
# scaler2 = Normalizer()
# test = scaler2.fit_transform(test)

# Model complexity
neig = np.arange(1, 5)
kfold = 10
train_accuracy = []
train_accuracy_sc = []
val_accuracy = []
val_accuracy_sc = []
bestKnn = None
bestAcc = 0.0
bestKnn_sc = None
bestAcc_sc = 0.0
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_sc = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(X_train,y_train)
    knn_sc.fit(X_train_sc, y_train)
    #train accuracy
    train_accuracy.append(knn.score(X_train, y_train))
    train_accuracy_sc.append(knn_sc.score(X_train_sc, y_train))
    # test accuracy
    val_accuracy.append(np.mean(cross_val_score(knn, X, y, cv=kfold)))
    val_accuracy_sc.append(np.mean(cross_val_score(knn_sc, X_sc, y, cv=kfold)))
    if np.mean(cross_val_score(knn_sc, X_sc, y, cv=kfold)) > bestAcc_sc:
        bestAcc_sc = np.mean(cross_val_score(knn_sc, X_sc, y, cv=kfold))
        bestKnn_sc = knn_sc
    if np.mean(cross_val_score(knn, X, y, cv=kfold)) > bestAcc:
        bestAcc = np.mean(cross_val_score(knn, X, y, cv=10))
        bestKnn = knn

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, val_accuracy, label = 'Validation Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('k value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()

print('Best Accuracy without feature scaling:', bestAcc)
print(bestKnn)

print('Best Accuracy with Standard Scaler:', bestAcc_sc)
print(bestKnn_sc)

# bestKnn.fit(X_norm, y)
submission = pd.DataFrame(bestKnn_sc.predict(scaler1.transform(test)))
print(submission)
submission.columns = ['Solution']
submission['Id'] = np.arange(1,submission.shape[0]+1)
# print(submission)
submission = submission[['Id', 'Solution']]
# submission = submission.drop(0, axis=1)
submission.to_csv('testLabels.csv', index=False)
print(submission)
