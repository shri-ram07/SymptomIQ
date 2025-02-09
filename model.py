import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

data = pd.read_csv("assets/datasets/Main.csv")
features = data.drop(["prognosis" ],axis=1)
features = np.array(features)
target = data.prognosis


# Sequential encoding using pandas.factorize()
data['target_encoded'] = pd.factorize(target)[0]

target = np.array(data.target_encoded).reshape(-1,1)
target = to_categorical(target)

# Ensure the dataset has exactly 5000 rows




model = Sequential()
model.add(Dense(64, activation='relu', input_dim=len(features[0])))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(41, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(features, target, epochs=100, batch_size=100, verbose=1)
score = model.evaluate(features, target, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("assets/saved_models")