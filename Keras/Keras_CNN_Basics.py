# -*- coding: utf-8 -*-
"""
@author: Divyansh Choubisa
Ways of Implementing CNN in Keras 
"""

#Necessary Imports
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#1st Way of Implementing a CNN in Keras
model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),            
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
])

#2nd Way
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compiling The Model
# optimizer = keras.optimizers.SGD(learning_rate=0.001) #Selecting an optimizer and setting a lerning rate
model.compile(loss="binary_crossentropy" , optimizer="adam", metrics=['accuracy'])

#Train Model
model.fit(X_train, Y_train, epochs=100, batch_size=32) #Feed the input_data

#Evaluate The Model
model.evalute(x_test, y_test)

