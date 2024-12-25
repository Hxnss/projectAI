import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pickle
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# [Previous code remains the same until show_predictions function]

def show_predictions(ticker):
    st.header("Stock Predictions")
    
    try:
        # Load models
        my_model = pickle.load(open('./model/my_MLR.pickle', 'rb'))
        model_teknikal = pickle.load(open('./model/weekly_model.pickle', 'rb'))
        
        # Get stock data
        saham = yf.Ticker(ticker)
        
        # Calculate fundamental prediction
        # Get historical data for quarterly prediction
        daftar_harga = saham.history(start='2024-07-01', end='2024-11-01', interval='1mo')
        daftar_harga.reset_index(inplace=True)
        q_harga = []
        for i in range(0, len(daftar_harga), 3):
            q_harga.append(daftar_harga['Close'][i])
        q_harga.reverse()

        # Get financial data
        neraca = saham.quarterly_balance_sheet
        income = saham.quarterly_income_stmt

        # Calculate financial metrics
        net_income = income.loc['Net Income'].iloc[0:2]
        saham_beredar = neraca.loc['Ordinary Shares Number'].iloc[0:2]
        ekuitas = neraca.loc['Total Equity Gross Minority Interest'].iloc[:2]
        aset = neraca.loc['Total Assets'].iloc[:2]
        
        # Calculate ratios
        eps = net_income/saham_beredar
        bpvs = ekuitas/saham_beredar
        pb = q_harga/bpvs
        roa = net_income/aset

        # Calculate differences
        diff_net_income = (net_income.iloc[0] - net_income.iloc[1])/net_income.iloc[1]
        diff_eps = (eps.iloc[0] - eps.iloc[1])/eps.iloc[1]
        diff_pb = (pb.iloc[0] - pb.iloc[1])/pb.iloc[1]
        diff_roa = (roa.iloc[0] - roa.iloc[1])/roa.iloc[1]

        # Make fundamental prediction
        q_prediction = my_model.predict([[diff_net_income, diff_eps, diff_pb, diff_roa]])

        # Calculate technical prediction
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        harga_saham = saham.history(start=start_date, end=end_date)
        
        # Calculate technical indicators
        harga_saham['SMA_10'] = harga_saham['Close'].rolling(window=10).mean()
        harga_saham['SMA_50'] = harga_saham['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = harga_saham['Close'].diff(1)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        harga_saham['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        harga_saham['SMA_20'] = harga_saham['Close'].rolling(window=20).mean()
        harga_saham['BB_Upper'] = harga_saham['SMA_20'] + 2 * harga_saham['Close'].rolling(window=20).std()
        harga_saham['BB_Lower'] = harga_saham['SMA_20'] - 2 * harga_saham['Close'].rolling(window=20).std()
        
        # Calculate ATR
        high_low = harga_saham['High'] - harga_saham['Low']
        high_close_prev = np.abs(harga_saham['High'] - harga_saham['Close'].shift(1))
        low_close_prev = np.abs(harga_saham['Low'] - harga_saham['Close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        harga_saham['ATR'] = true_range.rolling(window=14).mean()
        
        # Calculate Log Return
        harga_saham['Log_Return'] = np.log(harga_saham['Close'] / harga_saham['Close'].shift(1))

        # Make technical prediction
        hasil_prediksi = model_teknikal.predict([harga_saham.iloc[-1][['SMA_10', 'SMA_50', 'Volume', 'RSI', 'BB_Upper', 'BB_Lower', 'ATR', 'Log_Return']].to_list()])
        
        # Calculate predicted prices
        selisih = (float(hasil_prediksi[0]) - harga_saham['Close'].iloc[-1])/5
        l_pred = []
        for s in range(1, 6):
            l_pred.append(harga_saham['Close'].iloc[-1] + (s*selisih))
        
        # Create prediction dates
        prediksi_tanggal = [end_date + timedelta(days=i) for i in range(1, 6)]
        
        # Create prediction dataframe
        data_prediksi = pd.DataFrame({
            'Date': prediksi_tanggal,
            'Close': l_pred
        })
        
        # Display predictions
        st.subheader("Fundamental Analysis Prediction")
        prediction_value = round(float(q_prediction[0]), 2)
        color = "green" if prediction_value > 0 else "red"
        st.markdown(f"<h3 style='color: {color};'>Predicted quarterly change: {prediction_value}%</h3>", unsafe_allow_html=True)
        
        st.subheader("Technical Analysis Prediction")
        
        # Prepare data for chart
        data_harga = harga_saham[['Close']].iloc[:]
        data_harga.index = data_harga.index.tz_localize(None)
        data_prediksi['Date'] = pd.to_datetime(data_prediksi['Date'])
        data_prediksi.set_index('Date', inplace=True)
        data_final = pd.concat([data_harga, data_prediksi])
        
        # Create the prediction chart
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data_harga.index,
            y=data_harga['Close'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add prediction data
        fig.add_trace(go.Scatter(
            x=data_prediksi.index,
            y=data_prediksi['Close'],
            mode='lines',
            name='Prediction',
            line=dict(color='orange', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Stock Price Prediction for {ticker}',
            xaxis_title='Date',
            yaxis_title='Price',
            showlegend=True,
            template='plotly_white'
        )
        
        # Add separator line
        fig.add_vline(x=data_harga.index[-1], line_dash="dash", line_color="red")
        
        # Display the chart
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("Please make sure all required data is available and models are properly loaded.")

# [Rest of the code remains the same]

if __name__ == '__main__':
    main()