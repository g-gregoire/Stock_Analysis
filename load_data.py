# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:27:12 2020

@author: ns_10
"""

import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import stock_utils
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'


stock_names = ["aapl", "ibm", "amd", "hpq", "xrx", "msft"]

for i in range(len(stock_names)):
    stock_names[i] +=".us.txt"

# data_path = r'.\Dataset\Stocks'
data_path = os.path.expanduser("~/Documents/Github/Dataset/Stocks") #for Mac
# filenames = [os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.txt') and os.path.getsize(os.path.join(data_path,f)) > 0]
# filenames = random.sample(filenames,5)
filenames = [os.path.join(data_path, f) for f in stock_names]

data = []
for file in filenames:
    
    df = pd.read_csv(file)
    
    df['Label'] = file.split('\\')[-1].split('.')[0]
    df['Date'] = pd.to_datetime(df['Date'])
    data.append(df)
    

#%% plot sample of data    
r = lambda: random.randint(0,255)
traces = []

for df in data:
    clr_str = 'rgb('+str(r())+','+str(r())+','+str(r())+')'
    
    df = df.sort_values('Date')
    label = df['Label'].iloc[0]
    
#     trace = go.Scattergl(
#         x=df['Date'],
#         y=df['Close'],
#         mode='lines',
#         line=dict(color=clr_str),
#         name=label)
#     traces.append(trace)
    
# layout = go.Layout(title='Sample plot of Stocks')
# fig = go.Figure(data=traces,layout=layout)
# fig.update_xaxes(title_text='Date')
# fig.update_yaxes(title_text='Close Price')
# fig.show()

#%% create windows 
df = data[0] #takes the first stock for what we will predict
window_len = 10 #number of days of closes for window

#split train and test set
split_date = list(data[0]['Date'][-(2*window_len+1):])[0]
training_set, test_set = df[df['Date'] < split_date], df[df['Date'] >= split_date]
training_set = training_set.drop(['Date','Label','OpenInt'],1)
test_set = test_set.drop(['Date','Label','OpenInt'],1)



#create training windows
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    
    for col in list(temp_set):
        temp_set[col] = temp_set[col]/temp_set[col].iloc[0]-1
        
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

#create testing windows
LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    
    for col in list(temp_set):
        temp_set[col] = temp_set[col]/temp_set[col].iloc[0]-1
    
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1

LSTM_test_inputs = [np.array(LSTM_test_input) for LSTM_test_input in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

#%% build and train model architecture
nn_model = stock_utils.build_model(LSTM_training_inputs,output_size=1,neurons=32)

#train model
nn_history = nn_model.fit(LSTM_training_inputs,LSTM_training_outputs,epochs=5,batch_size=1,verbose=2,shuffle=True)
LSTM_test_predictions = nn_model.predict(LSTM_test_inputs)

#plot predictions
plt.figure()
plt.plot(LSTM_test_outputs,label='actual')
plt.plot(LSTM_test_predictions,label='predicted')
plt.legend()
plt.title('Predicted and true outputs from LSTM Model: '+df['Label'][0])
plt.ylabel('Closing Price')
plt.xlabel('Time')

MAE = mean_absolute_error(LSTM_test_outputs,LSTM_test_predictions)
print('MAE is: {}'.format(MAE))

#predict full sequence
predictions = stock_utils.predict_sequence_full(nn_model,LSTM_test_inputs,10)

plt.figure()
plt.plot(predictions,label='predicted')
plt.plot(LSTM_test_outputs,label='actual')
plt.legend()
plt.title('Full Sequence Prediction: '+df['Label'][0])
plt.ylabel('Closing Price')
plt.xlabel('Time')

MAE = mean_absolute_error(LSTM_test_outputs,predictions)
print('Full Sequence MAE is: {}'.format(MAE))