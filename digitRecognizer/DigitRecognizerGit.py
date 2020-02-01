# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:02:43 2018

@author: matth
"""

"""
IN ORDER TO ENHANCE THE MODEL YOU CAN ADD A CONV LAYER OR MORE. YOU CAN ALSO MODULATE THE TARGET SIZE OF THE IMAGE YOU ARE WORKING ON. (ATM ON 64X64 PIXELS)
"""
#Build CNN
import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop




# Read training and test data files
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

y_train = train["label"].values
X_train = train.drop(labels = ["label"],axis = 1) 

#Reshape and normalize training data

X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)





#LabelEncoder
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)

# Split the train and the validation set for the fitting
X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_train, y_train, test_size = 0.1)




#Init CNN
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5, 5) , input_shape=(28,28,1),padding = 'Same',activation= 'relu'))
#model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters = 32,kernel_size = (5, 5) ,padding = 'Same',activation= 'relu' ))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters =64,kernel_size =(3, 3), padding = 'Same',activation= 'relu' ))
#model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters =64,kernel_size =(3, 3),padding = 'Same',activation= 'relu' ))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation= 'relu' ))
model.add(Dropout(0.3))
model.add(Dense(10, activation= 'softmax' ))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  # Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# Add generate image to prevent bias
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#Fit
datagen.fit(X_trainS)

epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

model.fit_generator(
        datagen.flow(X_trainS,y_trainS, batch_size=batch_size),
        steps_per_epoch=X_train.shape[0] // batch_size
        ,callbacks=[learning_rate_reduction],
        epochs=epochs,
        validation_data=(X_testS,y_testS))



# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagenGit.csv",index=False)