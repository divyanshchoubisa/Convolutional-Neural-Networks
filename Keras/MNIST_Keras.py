# -*- coding: utf-8 -*-
"""
@author: Divyansh Choubisa
Feed-Forward Neural Network, MNIST TRAINING AND EVALUATION KERAS
"""

import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1) 

x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()  # a basic feed-forward model

model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=30)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate 

print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy