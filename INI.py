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
    metrics = {}
    
    try:
        # Net Income
        net_income_current = data_laba.loc['Net Income'].iloc[0]
        net_income_prev = data_laba.loc['Net Income'].iloc[1]
        metrics['Net Income'] = {
            'value': net_income_current,
            'diff': ((net_income_current - net_income_prev) / net_income_prev * 100)
        }
        
        # Total Assets
        assets_current = data_neraca.loc['Total Assets'].iloc[0]
        assets_prev = data_neraca.loc['Total Assets'].iloc[1]
        metrics['Total Assets'] = {
            'value': assets_current,
            'diff': ((assets_current - assets_prev) / assets_prev * 100)
        }
        
        # Total Liabilities
        liabilities_current = data_neraca.loc['Total Liabilities Net Minority Interest'].iloc[0]
        liabilities_prev = data_neraca.loc['Total Liabilities Net Minority Interest'].iloc[1]
        metrics['Total Liabilities'] = {
            'value': liabilities_current,
            'diff': ((liabilities_current - liabilities_prev) / liabilities_prev * 100)
        }
        
        # Total Equity
        equity_current = data_neraca.loc['Total Equity Gross Minority Interest'].iloc[0]
        equity_prev = data_neraca.loc['Total Equity Gross Minority Interest'].iloc[1]
        metrics['Total Equity'] = {
            'value': equity_current,
            'diff': ((equity_current - equity_prev) / equity_prev * 100)
        }
        
        # Revenue
        revenue_current = data_laba.loc['Operating Revenue'].iloc[0]
        revenue_prev = data_laba.loc['Operating Revenue'].iloc[1]
        metrics['Revenue'] = {
            'value': revenue_current,
            'diff': ((revenue_current - revenue_prev) / revenue_prev * 100)
        }
        
        # Cash and Cash Equivalents
        cash_current = data_neraca.loc['Cash And Cash Equivalents'].iloc[0]
        cash_prev = data_neraca.loc['Cash And Cash Equivalents'].iloc[1]
        metrics['Cash'] = {
            'value': cash_current,
            'diff': ((cash_current - cash_prev) / cash_prev * 100)
        }
        
        # Capital Expenditure
        capex_current = data_arus.loc['Capital Expenditure'].iloc[0]
        capex_prev = data_arus.loc['Capital Expenditure'].iloc[1]
        metrics['Capital Expenditure'] = {
            'value': capex_current,
            'diff': ((capex_current - capex_prev) / capex_prev * 100)
        }
        
        # Outstanding Shares
        shares = data_neraca.loc['Ordinary Shares Number'].iloc[0]
        metrics['Outstanding Shares'] = {
            'value': shares,
            'diff': None  # No difference calculation for shares
        }
        
    except Exception as e:
        st.error(f"Error calculating financial metrics: {str(e)}")
        
    return metrics

def show_financial_details(ticker):
    st.header("Financial Details")
    
    # Get stock data
    saham = yf.Ticker(ticker)
    
    # Get financial statements
    data_neraca = saham.quarterly_balance_sheet
    data_laba = saham.quarterly_income_stmt
    data_arus = saham.quarterly_cash_flow
    
    # Calculate metrics
    metrics = calculate_financial_metrics(data_neraca, data_laba, data_arus)
    
    # Create three columns for better organization
    col1, col2, col3 = st.columns(3)
    
    # Helper function to format values
    def format_value(value):
        if abs(value) >= 1e9:
            return f"{value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.2f}M"
        else:
            return f"{value:,.2f}"
    
    # Helper function to create metric display with proper formatting
    def display_metric(col, title, metric_data):
        if metric_data and 'value' in metric_data:
            formatted_value = format_value(metric_data['value'])
            delta = f"{metric_data['diff']:.2f}%" if metric_data.get('diff') is not None else None
            col.metric(
                title,
                formatted_value,
                delta,
                delta_color="normal"
            )
    
    # Distribute metrics across columns
    metrics_per_column = len(metrics) // 3 + (len(metrics) % 3 > 0)
    
    # First column
    with col1:
        st.markdown("##### Balance Sheet Metrics")
        display_metric(col1, "Total Assets", metrics.get('Total Assets'))
        display_metric(col1, "Total Liabilities", metrics.get('Total Liabilities'))
        display_metric(col1, "Total Equity", metrics.get('Total Equity'))
    
    # Second column
    with col2:
        st.markdown("##### Income Statement Metrics")
        display_metric(col2, "Revenue", metrics.get('Revenue'))
        display_metric(col2, "Net Income", metrics.get('Net Income'))
        display_metric(col2, "Outstanding Shares", metrics.get('Outstanding Shares'))
    
    # Third column
    with col3:
        st.markdown("##### Cash Flow Metrics")
        display_metric(col3, "Cash & Equivalents", metrics.get('Cash'))
        display_metric(col3, "Capital Expenditure", metrics.get('Capital Expenditure'))
    
    # Show stock price chart
    st.markdown("### Stock Price Chart")
    show_stock_chart(ticker)

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