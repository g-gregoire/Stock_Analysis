# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:09:42 2020

@author: ns_10
"""

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout

def build_model(inputs,output_size,neurons,activ='linear',dropout=0.1,loss='mae',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons,input_shape=(inputs.shape[1],inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size,activation=activ))
    
    model.compile(loss=loss,optimizer=optimizer)
    return model

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted