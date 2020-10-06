#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:09:13 2020

@author: chenyuchen
"""

from model_builder import AbstractModelBuilder
#from keras.layers import concatenate
# Import Keras libraries and packages
from keras.models import Sequential  #NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks


class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

            
       
        def buildModel(self):
            #model 34 
            #layers: the more the better , units: twice of inputs or more
            #add pruning
            model = Sequential()  
            model.add(Dense(128, input_shape=(61,), activation = 'relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            return model
