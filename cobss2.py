import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pickle
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Set page config
st.set_page_config(page_title="KKB Project", layout="wide")

# Custom CSS to maintain similar styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        margin: 10px 0;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .main {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)

class MLR:
    def __init__(self, data):
        X = [list(i[:-1]) for i in data]
        for xi in X:
            xi.insert(0, 1)
        X = np.array(X)
        Y = np.array([[i[-1]] for i in data])
        y_bar = sum(Y)/len(Y)
        XT = X.T
        XTX = np.dot(XT, X)
        XTX_inv = np.linalg.inv(XTX)
        XTY = np.dot(XT, Y)
        self.B = np.dot(XTX_inv, XTY)
        self.SST = sum([(Y[i][0] - y_bar)**2 for i in range(len(X))])
        y_pred = []
        for i in range(len(X)):
            t_pred = self.B[0][0]
            for b in range(1, len(self.B)):
                px = X[i][b-1]*self.B[b][0]
                t_pred += px
            y_pred.append(t_pred)
        self.SSE = sum([(Y[i][0] - y_pred[i])**2 for i in range(len(X))])
        self.R2 = 1 - (self.SSE/self.SST)

    def predict(self, X):
        prediction = []
        for i in range(len(X)):
            t_pred = self.B[0][0]
            for b in range(1, len(self.B)):
                px = X[i][b-1]*self.B[b][0]
                t_pred += px
            prediction.append(t_pred)
        return prediction

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Homepage", "About Project", "Stock Analysis"]
    )

    if page == "Homepage":
        show_homepage()
    elif page == "About Project":
        show_about()
    elif page == "Stock Analysis":
        show_stock_analysis()

def show_homepage():
    st.title("Welcome To KKB Project")
    st.header("Portfolio Optimization")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("About Project"):
            st.session_state.page = "About Project"
            st.experimental_rerun()
    with col2:
        if st.button("Try This Project!"):
            st.session_state.page = "Stock Analysis"
            st.experimental_rerun()

def show_about():
    st.title("About This Project")
    # Add project description here

def show_stock_analysis():
    st.title("Stock Analysis")
    
    # List of stocks
    stocks = {
        "BBRI.JK": "Bank Rakyat Indonesia",
        "BMRI.JK": "Bank Mandiri",
        "BBCA.JK": "Bank Central Asia",
        "ASII.JK": "Astra International",
        "UNVR.JK": "Unilever Indonesia",
        "TLKM.JK": "Telekomunikasi Indonesia"
    }
    
    # Stock selection
    selected_ticker = st.selectbox("Select a stock:", list(stocks.keys()))
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Company Details", "Financial Details", "Predictions"])
    
    with tab1:
        show_company_details(selected_ticker)
    
    with tab2:
        show_financial_details(selected_ticker)
    
    with tab3:
        show_predictions(selected_ticker)

def show_company_details(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    st.header("Company Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Company Name", info.get('longName', 'N/A'))
        st.metric("Sector", info.get('sector', 'N/A'))
        st.metric("Industry", info.get('industry', 'N/A'))
    
    with col2:
        st.metric("Market Cap", info.get('marketCap', 'N/A'))
        st.metric("Previous Close", info.get('previousClose', 'N/A'))

def show_financial_details(ticker):
    saham = yf.Ticker(ticker)
    
    # Get financial data
    data_neraca = saham.quarterly_balance_sheet
    data_laba = saham.quarterly_income_stmt
    data_arus = saham.quarterly_cash_flow
    
    # Calculate metrics
    metrics = calculate_financial_metrics(data_neraca, data_laba, data_arus)
    
    # Display metrics
    st.header("Financial Details")
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        for key in list(metrics.keys())[:len(metrics)//2]:
            st.metric(
                key,
                f"{metrics[key]['value']:,.2f}",
                f"{metrics[key]['diff']:.2f}%" if 'diff' in metrics[key] else None
            )
    
    with col2:
        for key in list(metrics.keys())[len(metrics)//2:]:
            st.metric(
                key,
                f"{metrics[key]['value']:,.2f}",
                f"{metrics[key]['diff']:.2f}%" if 'diff' in metrics[key] else None
            )

    # Show stock price chart
    show_stock_chart(ticker)

def calculate_financial_metrics(data_neraca, data_laba, data_arus):
    # Extract and calculate all financial metrics
    metrics = {}
    
    # Basic metrics
    metrics['Net Income'] = {
        'value': data_laba.loc['Net Income'].iloc[0],
        'diff': ((data_laba.loc['Net Income'].iloc[0] - data_laba.loc['Net Income'].iloc[1]) / 
                data_laba.loc['Net Income'].iloc[1] * 100)
    }
    
    metrics['Total Assets'] = {
        'value': data_neraca.loc['Total Assets'].iloc[0],
        'diff': ((data_neraca.loc['Total Assets'].iloc[0] - data_neraca.loc['Total Assets'].iloc[1]) /
                data_neraca.loc['Total Assets'].iloc[1] * 100)
    }
    
    # Add more metrics as needed
    
    return metrics

def show_stock_chart(ticker):
    # Get stock data
    stock = yf.Ticker(ticker)
    data = stock.history(period='3mo')
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date'
    )
    
    st.plotly_chart(fig)

def show_predictions(ticker):
    st.header("Stock Predictions")
    
    # Load models
    try:
        my_model = pickle.load(open('./model/my_MLR.pickle', 'rb'))
        model_teknikal = pickle.load(open('./model/weekly_model.pickle', 'rb'))
        
        # Calculate predictions
        # Add your prediction logic here
        
        # Display predictions
        st.subheader("Fundamental Analysis Prediction")
        # Add fundamental prediction display
        
        st.subheader("Technical Analysis Prediction")
        # Add technical prediction display
        
    except Exception as e:
        st.error("Error loading prediction models. Please check the model files.")

if __name__ == '__main__':
    main()