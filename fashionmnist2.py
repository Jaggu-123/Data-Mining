import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

data = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_test = x_test/255.0
x_train = x_train/255.0

print(x_test.shape)
print(x_train.shape)

# cnn expects 3 dimensional input
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

K = len(set(y_train))

# Building the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D

i = Input(shape=(28,28,1))
x = Conv2D(filters=32, kernel_size=(3,3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(units=100)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

model = Model(i, x)

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=8)

print("Returned:", r)

# print the available keys
# should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
print(r.history.keys())

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()