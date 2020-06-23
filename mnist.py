import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    print('Normalized Confusion Matrix')
  else:
    print('Confusion Matrix without Normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.0
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True Label')
  plt.xlabel("Predicted label")
  plt.show()

train = pd.read_csv("digit-recognizer/train.csv")
test = pd.read_csv("digit-recognizer/test.csv")
trainLabel = train['label']
train = train.drop(labels=['label'], axis=1)

train = train/255.0
test = test/255.0

train = train.values.reshape(-1,28,28)
test = test.values.reshape(-1,28,28)
trainLabel = trainLabel.values

from sklearn.model_selection import train_test_split
trainX, trainx, trainLabelX, trainLabelx = train_test_split(train, trainLabel, test_size=0.1, random_state=0)

import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(trainX.shape)
print(trainLabelX.shape)
r = model.fit(train, trainLabel, epochs=10)
# print(model.evaluate(trainx, trainLabelx))

p_test = model.predict(trainx).argmax(axis = 1)
cm = confusion_matrix(trainLabelx, p_test)
plot_confusion_matrix(cm, list(range(10)))

# predict results
results = model.predict(test)
print(results)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("digit-recognizer/mnist_datagen.csv",index=False)
