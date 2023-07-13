# import the libraries
import datetime
from math import sqrt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import tensorflow as tf
keras = tf.keras
import yfinance as yf


import warnings 
warnings.filterwarnings("ignore")


# reading the dataset and printing the top 5 values
start_date = datetime.date(2013,1,1)
end_date = datetime.datetime.today()
tickerSymbol = 'INFY'

stock = yf.download(tickerSymbol,start = start_date,end = end_date,progress = False)
df = stock.reset_index()['Close']

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df).reshape(-1,1))

train_size = int(len(df)*0.70)
test_size = len(df) - train_size
train_data,test_data = df1[0:train_size,:],df1[train_size:len(df),:1]

# Convert an array of values into a dataset matrix
def create_dataset(dataset,timestamp = 1):
    X, y = [], []
    for i in range(len(dataset)-timestamp-1):
        a = dataset[i:(i+timestamp),0]
        X.append(a)
        y.append(dataset[i + timestamp, 0])
    return np.array(X),np.array(y)

timestamp = 100
X_train , y_train = create_dataset(train_data,timestamp)
X_test , y_test = create_dataset(test_data,timestamp)

#reshape input to be [samples,time steps,features] required for lstm
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer = 'adam',loss='mean_squared_error')
model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs=100,batch_size=64,verbose=2)

model.save('lstm_model.h5')