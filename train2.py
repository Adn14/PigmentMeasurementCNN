# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:38:09 2019

@author: Adn
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import csv
from csv import reader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_csv(filename):
    
    data= list()
    with open(filename, 'r', encoding='utf-8-sig') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data

    path = ""
    arrayDirBox = os.listdir('box_inpaint')
    arrayDirRoom = os.listdir('room_inpaint')


    all_images = []

    ##load image data
    #for i in range(len(arrayDirBox)):
    #    image = cv2.imread("box_cropped/"+arrayDirBox[i], cv2.IMREAD_UNCHANGED)
    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #    image = cv2.resize(image, (34, 34))
    #    all_images.append(image)
        

    for i in range(len(arrayDirRoom)):
        image = cv2.imread("room_inpaint/"+arrayDirRoom[i], cv2.IMREAD_UNCHANGED)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        image = cv2.resize(image, (34, 34))
        image2 = cv2.resize(image2, (34, 34))
        
        image = np.array(image)
        image2 = np.array(image2)
        
        image = np.concatenate((image, image2), axis=2)
        
        all_images.append(image)

    print(all_images[0].shape)
    shape=all_images[0].shape
    X_data = np.array(all_images)
    X_data = X_data.astype(np.floating)
    X_data /= 255
    print(X_data.shape)

    ##load target data
    filename = 'data_target.csv'
    data = load_csv(filename)
    data = np.array(data)

    dataset = list()

    #for i in range(len(arrayDirBox)):
    #    for j in range(len(data)):
    #        if( data[j, 0] == os.path.splitext(arrayDirBox[i])[0]):
    #            dataset.append(data[j, 1:5])


    for i in range(len(arrayDirRoom)):
        for j in range(len(data)):
            if( data[j, 0] == os.path.splitext(arrayDirRoom[i])[0]):
                dataset.append(data[j, 1:5])

                
    Y_data = np.array(dataset)
    print(Y_data.shape)

    #shuffle data
    X_data, Y_data = shuffle(X_data, Y_data, random_state=2)

    #split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=2)

    print(X_train.shape, Y_train.shape)


    #print(X_data[0].ndim)


    datagen = ImageDataGenerator(rotation_range=40,
                                 zoom_range=0.2, 
                                 fill_mode='nearest')

    model = Sequential()
    model.add(Conv2D(32, 
                     kernel_size=(3,3),
                     strides=(2, 2),
                     activation='relu',
                     input_shape=shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(90, activation='relu'))
    model.add(Dropout(0.03))
    model.add(Dense(3))
    model.add(LeakyReLU(alpha=0.7))
    model.compile(loss="mean_squared_error",
                  optimizer=Nadam(),
                  metrics=['mae'])
        
        #number of epoch and batch size
    EPOCHS = 200
    BS = 29
        
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= BS), 
                        validation_data=(X_test, Y_test), 
                        
                        epochs = EPOCHS)

    #save
    print("saving model")
    model_json = model.to_json()
    with open("model_inpaint_lab_ycbcr.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model_inpaint_lab_ycbcr.h5")
    print("finished : model saved")    

    # Plot training & validation loss values
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model loss')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    print(history.history['mean_absolute_error'][-1])

    with open('score.csv', mode='a', newline='') as score:
        score_writer = csv.writer(score, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        score_writer.writerow([history.history['loss'][-1], history.history['mean_absolute_error'][-1], 
                               history.history['val_loss'][-1], history.history['val_mean_absolute_error'][-1]])
        
