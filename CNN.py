import tensorflow as tf

from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt
#import pandas as pd
#import numpy as np
#from matplotlib import pyplot
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from google.colab import files

train= pd.read_csv("trainset.csv")

train_y=train.iloc[:,0]
train_x= train.iloc[:,train.columns!='label']
train_x_numpy=train_x.to_numpy()

X_train=train_x_numpy.reshape(28000,28,28)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

trainY = np_utils.to_categorical(train_y)
# convert from integers to floats
X_train = X_train.astype('float32')

# normalize to range 0-1
X_train = X_train / 255.0


model = models.Sequential()

model.add(layers.Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(padding="same"))

model.add(layers.Conv2D(64,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(layers.MaxPooling2D(padding="same"))

model.add(layers.Flatten())

model.add(layers.Dense(1024,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10,activation="sigmoid"))


model.summary()



model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(X_train
                                                    , trainY
                                                    , test_size=0.20
                                                    , shuffle=True
                                                    , random_state=32
                                                   )
												   
												   
												   
history = model.fit(X_train, y_train, epochs=5, batch_size=100, validation_data=(X_test, y_test), verbose=2)
		# evaluate model
_, acc = model.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))



test= pd.read_csv("testset.csv")

test_x_numpy=test.to_numpy()
print(test_x_numpy)

X_test=test_x_numpy.reshape(14000,28,28)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# convert from integers to floats
X_test = X_test.astype('float32')

# normalize to range 0-1
X_test = X_test / 255.0


output=model.predict(X_test)



index = []
for i in range(0,14000):
  index.append(np.argmax(output[i], axis=0))
df = pd.DataFrame(index)
df.columns = ['Label']
df.index = np.arange(1, len(df)+1)
df.to_csv("results.csv", index=True)
files.download("results.csv")
print(df)