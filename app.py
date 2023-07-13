# IMPORT THE LIBRARIES
import streamlit as st
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import tensorflow as tf
keras = tf.keras
from keras.models import load_model

import warnings 
warnings.filterwarnings("ignore")

st.sidebar.title("Stock Price Prediction System")

print('\n')

start_date = datetime.date(2018,1,1)
end_date = datetime.datetime.today()
tickerSymbol = ' '

tickerSymbol = st.sidebar.text_input("Enter Ticker Symbol:")

try:

    stock = yf.download(tickerSymbol,start = start_date,end = end_date,progress = False)

    st.subheader(f'{tickerSymbol} STOCK PRICE DATA')

    st.write(stock)

    # Data Visualisation

    chart = st.sidebar.selectbox("Graph:",("Select","Line","Area","Candle Sticks"))


    st.subheader('Data Visualisation')
    if chart == 'Line':
        st.markdown('Line Chart')
        st.line_chart(stock['Close'],use_container_width=True)
        
    elif chart == 'Area':
        st.markdown('Area Chart')
        st.area_chart(stock['Close'], use_container_width=True)
        
    elif chart == 'Candle Sticks':
        st.markdown('Candle Stick')
        figure = go.Figure(data=[go.Candlestick(x=stock.index,
                                                open=stock["Open"], high=stock["High"],
                                                low=stock["Low"], close=stock["Close"])])
        # figure.update_layout( xaxis_rangeslider_visible=True)
        st.plotly_chart(figure)

    else:
        st.write('CHOOSE AN APPROPRIATE OPTION')


    # Forecasting

    no_days = st.sidebar.number_input('No. of days for Forecasting:')

    st.subheader(f'Forecasting for {no_days} days')

    df = stock.reset_index()['Close']
    model = load_model('D:\\RealtimeStockPrice\\lstm_model.h5')


    def fun(no_days):
        scaler = MinMaxScaler(feature_range=(0,1))
        df1 = scaler.fit_transform(np.array(df).reshape(-1,1))

        train_size = int(len(df)*0.70)
        test_size = len(df) - train_size
        train_data,test_data = df1[0:train_size,:],df1[train_size:len(df),:1]
        x_input = test_data[test_size-100:].reshape(1,-1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output=[]
        n_steps = 100
        i = 0
        while(i<=no_days):
            if(len(temp_input)>100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1,n_steps,1))
                yhat = model.predict(x_input,verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i+1
            else:
                x_input = x_input.reshape((1,n_steps,1))
                yhat = model.predict(x_input,verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i+1

        pred = scaler.inverse_transform(lst_output)
        output = [element for innerList in pred for element in innerList] 
        day_new = np.arange(1,101)
        day_pred = np.arange(101,131)
        return output


    if tickerSymbol != ' ':
        output = fun(no_days)
        output = pd.DataFrame(output)
        df2 = pd.concat([df, output], ignore_index=True)
        df2.rename(columns = {0:'Close'}, inplace = True)
        df3 = df2[1200:]
        st.line_chart(df3,use_container_width=True)
        st.write(output)
    else:
        pass
except:
    pass
