import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Load model
@st.cache_resource
def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

# Halaman utama
st.title("Aplikasi Saham Interaktif")
menu = st.sidebar.selectbox("Menu", ["Homepage", "Tentang Proyek", "Detail Saham", "Keuangan", "Prediksi"])

if menu == "Homepage":
    st.header("Selamat Datang di Aplikasi Saham")
    st.write("Gunakan menu di samping untuk bernavigasi.")

elif menu == "Tentang Proyek":
    st.header("Tentang Proyek")
    st.write("Detail tentang proyek ini akan dijelaskan di sini.")

elif menu == "Detail Saham":
    ticker = st.text_input("Masukkan kode saham (contoh: AAPL):")
    if ticker:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get('longName', 'Nama perusahaan tidak tersedia')
        sector = info.get('sector', 'Sektor tidak tersedia')
        industry = info.get('industry', 'Industri tidak tersedia')
        market_cap = info.get('marketCap', 'Market cap tidak tersedia')
        previous_close = info.get('previousClose', 'Harga penutupan sebelumnya tidak tersedia')

        st.write(f"**Nama Perusahaan:** {company_name}")
        st.write(f"**Sektor:** {sector}")
        st.write(f"**Industri:** {industry}")
        st.write(f"**Market Cap:** {market_cap}")
        st.write(f"**Harga Penutupan Sebelumnya:** {previous_close}")

elif menu == "Keuangan":
    ticker = st.text_input("Masukkan kode saham (contoh: AAPL):")
    if ticker:
        saham = yf.Ticker(ticker)
        data_harga_saham = saham.history(period='3mo')
        dates = data_harga_saham.index.strftime('%Y-%m-%d').tolist()
        closing_prices = data_harga_saham['Close'].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=closing_prices, mode='lines', name='Harga Penutupan'))
        fig.update_layout(
            title=f'Harga Saham {ticker} (3 Bulan Terakhir)',
            xaxis_title='Tanggal',
            yaxis_title='Harga (IDR)',
            template='plotly_white'
        )

        st.plotly_chart(fig)

elif menu == "Prediksi":
    ticker = st.text_input("Masukkan kode saham untuk prediksi (contoh: AAPL):")
    if ticker:
        my_model = load_model('./model/my_MLR.pickle')

        saham = yf.Ticker(ticker)
        daftar_harga = saham.history(start='2024-07-01', end='2024-11-01', interval='1mo')
        daftar_harga.reset_index(inplace=True)
        q_harga = []
        for i in range(0, len(daftar_harga), 3):
            q_harga.append(daftar_harga['Close'][i])
        q_harga.reverse()

        neraca = saham.quarterly_balance_sheet
        income = saham.quarterly_income_stmt

        net_income = income.loc['Net Income'].iloc[0:2]
        saham_beredar = neraca.loc['Ordinary Shares Number'].iloc[0:2]
        ekuitas = neraca.loc['Total Equity Gross Minority Interest'].iloc[:2]
        aset = neraca.loc['Total Assets'].iloc[:2]
        eps = net_income / saham_beredar
        bpvs = ekuitas / saham_beredar
        pb = q_harga / bpvs
        roa = net_income / aset

        diff_net_income = (net_income.iloc[0] - net_income.iloc[1]) / net_income.iloc[1]
        diff_eps = (eps.iloc[0] - eps.iloc[1]) / eps.iloc[1]
        diff_pb = (pb.iloc[0] - pb.iloc[1]) / pb.iloc[1]
        diff_roa = (roa.iloc[0] - roa.iloc[1]) / roa.iloc[1]

        q_prediction = my_model.predict([[diff_net_income, diff_eps, diff_pb, diff_roa]])

        st.write(f"**Prediksi Harga Saham Mendatang:** {round(float(q_prediction[0]), 2)}")
