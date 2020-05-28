#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:23:52 2020

@author: gregoireg
"""

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#print(tf.__version__) 

#%%
stock_names = ["aapl", "ibm", "amd", "intc", "msft"]

for i in range(len(stock_names)):
    stock_names[i] +=".us.txt"

data_path = r'\.\Dataset\Stocks'
print(os.listdir(data_path))
#%%
#data_path = os.path.expanduser("~/Documents/Github/Dataset/Stocks") #for Mac
filenames = [os.path.join(data_path, f) for f in stock_names]

data = []
for file in filenames:
    
    df = pd.read_csv(file)
    
    df['Label'] = file.split('\\')[-1].split('.')[0]
    df['Date'] = pd.to_datetime(df['Date'])
    data.append(df)

## Windows
split_time = 500
window_size = 60#40
batch_size = 100
predict_size = 1 #The model fails if you make this >1, doesn't seem to be able to handle it
time_step = range(len(data[0]['Date']))
close_series = data[0]['Close']
close_train = close_series[-5000:-split_time] #, data[0]['Open'][-20:], data[0]['High'][-20:], data[0]['Low'][-20:]]
#print(close_train.size)
close_test = close_series[-split_time:]
#print(close_train.shape)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def create_ds(ds):
    dataset = tf.expand_dims(ds, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(window_size + predict_size, shift=1, drop_remainder= True)
    dataset = dataset.flat_map(lambda x: x.batch(window_size + predict_size))
    dataset = dataset.map(lambda x: (x[:-predict_size], x[-predict_size:]))
    dataset = dataset.shuffle(10)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def predict_ds(model, series, window_size):
    ds = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.window(window_size, shift=1, drop_remainder= True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1) 
    forecast = model.predict(ds)
    return forecast

#Create Tensorflow dataset object
dataset = create_ds(close_train)
    
#%% Model Creation & Fit

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

#lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#    lambda epoch: 1e-8 * 10**(epoch / 20))
#optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
#model.compile(loss=tf.keras.losses.Huber(),
#              optimizer=optimizer,
#              metrics=["mae"])
#history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

optimizer = tf.keras.optimizers.SGD(lr=8e-7, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
history = model.fit(dataset, epochs=200)

#%%
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 60])

#%% Prediction & Plotting

forecast = predict_ds(model, close_series, window_size)
#print(forecast[100])
forecast = forecast[-split_time:, -1, 0]
print(forecast)


plt.figure(figsize=(10, 6))
plot_series(time_step[-split_time:], close_test)
plot_series(time_step[-split_time:], forecast)

#%%
#tf.keras.metrics.mean_absolute_error(close_test, forecast).numpy()
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs


#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.title('Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])

plt.figure()

zoomed_loss = loss[100:]
zoomed_epochs = range(100,len(loss))


#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(zoomed_epochs, zoomed_loss, 'r')
plt.title('Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])

plt.figure()