import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD

import time

IMG_SIZE = 50
#loading the pickle object 
finaldf = pd.read_pickle('finaldf.pkl')  #path to the prepared .pkl file (prepared using data_preparation.py)

image_pixels = np.array(finaldf['image'])
image_pixels= [i for i in image_pixels]
gender_label = np.array(finaldf['gender'])
age_label = np.array(finaldf['age'])
race_label = np.array(finaldf['race'])

X = np.array(image_pixels).reshape(-1,IMG_SIZE,IMG_SIZE,1)/255.0
# training model for gender classification
y = np.array(gender_label)

MODEL_NAME = f"gender-cnn-64x3-dense-64x1-{int(time.time())}"
tensorboard = TensorBoard(log_dir = f'logs/{MODEL_NAME}')
model  = Sequential()

model.add(Conv2D(64,(3,3),input_shape = (IMG_SIZE,IMG_SIZE,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten())
model.add(Dense(64, activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = "binary_crossentropy",
             optimizer = 'adam',
             metrics = ['accuracy'])

model.fit(X , y ,batch_size = 32 ,validation_split = 0.2, epochs=20)  #fitting the model to our data
model.save(f"{MODEL_NAME}.h5")  #saving the model as .h5 file


# training the model for age classification
y = np.array(age_label)
y = tf.keras.utils.to_categorical(y, num_classes=9)


MODEL_NAME = f"age-cnn-64x3-dense-64x1-{int(time.time())}"
tensorboard = TensorBoard(log_dir = f'logs/{MODEL_NAME}')
model  = Sequential()

model.add(Conv2D(64,(3,3),input_shape = (IMG_SIZE,IMG_SIZE,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(9,activation = 'softmax'))

model.compile(loss = "categorical_crossentropy",
             optimizer = 'adam',
             metrics = ['accuracy'])

model.fit(X , y ,batch_size = 32 ,validation_split = 0.2, epochs=50)  #fitting the model to our data

model.save(f"{MODEL_NAME}.h5") # saving the model as .h5 file

# training model for race classification

y = np.array(race_label)
y = tf.keras.utils.to_categorical(y, num_classes=6)


MODEL_NAME = f"race-cnn-64x3-dense-64x1-{int(time.time())}"
tensorboard = TensorBoard(log_dir = f'logs/{MODEL_NAME}')
model  = Sequential()

model.add(Conv2D(64,(3,3),input_shape = (IMG_SIZE,IMG_SIZE,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(6,activation = 'softmax'))

model.compile(loss = "categorical_crossentropy",
             optimizer = 'adam',
             metrics = ['accuracy'])

model.fit(X , y ,batch_size = 32 ,validation_split = 0.2, epochs=50)  #fitting the model to our data


model.save(f"{MODEL_NAME}.h5") # saving the model as .h5 file