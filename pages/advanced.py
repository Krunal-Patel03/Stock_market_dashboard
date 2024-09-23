import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import talib as ta

st.set_page_config(page_title="Advanced_chart", layout="wide")

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


# Function to plot stock chart
def plot_stock_chart(stock_data, indicators, overlays, annotations, drawing_tool):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                  open=stock_data['Open'],
                                  high=stock_data['High'],
                                  low=stock_data['Low'],
                                  close=stock_data['Close'], name='Candlesticks'))
    

    # Add selected indicators
    for indicator in indicators:
        if indicator == 'SMA':
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50'))
        elif indicator == 'EMA':
            stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_20'], mode='lines', name='EMA 20'))
        elif indicator == 'RSI':
            stock_data['RSI'] =  ta.RSI(stock_data['Close'])
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='blue')))
            # Add horizontal lines at RSI levels
            fig.add_shape(type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70, line=dict(color="green", width=1, dash="dash"))
            fig.add_shape(type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30, line=dict(color="red", width=1, dash="dash"))
        elif indicator == 'MACD':
            macd, macd_signal, macd_hist = ta.MACD(stock_data['Close'])
            # Add MACD and Signal traces
            fig.add_trace(go.Scatter(x=stock_data.index, y=macd, mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=stock_data.index, y=macd_signal, mode='lines', name='Signal'))
            # Determine color for MACD histogram bars
            colors = ['red' if cl < 0 else 'green' for cl in macd_hist]
            # Add MACD Histogram bars
            fig.add_trace(go.Bar(x=stock_data.index, y=macd_hist, marker=dict(color=colors), name='MACD Histogram'))


    # Add overlays
    for overlay in overlays:
        if overlay == 'Bollinger Bands':
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
            std_dev = stock_data['Close'].rolling(window=20).std()
            stock_data['Upper_BB'] = stock_data['SMA_20'] + 2 * std_dev
            stock_data['Lower_BB'] = stock_data['SMA_20'] - 2 * std_dev
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_BB'], mode='lines', name='Upper BB'))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_BB'], mode='lines', name='Lower BB'))

    # Add annotations
    for annotation in annotations:
        fig.add_annotation(x=stock_data.index[-1], y=stock_data['Close'][-1], text=annotation, showarrow=True,
                           arrowhead=1, arrowcolor="blue")

    # Drawing tool - trend line
    if drawing_tool:
        fig.update_layout(
            dragmode='drawline',
            newshape=dict(
                line_color="cyan",
                opacity=0.5,
            )
        )

    fig.update_layout(title='Stock Chart', xaxis_rangeslider_visible=True,height=700)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.sidebar.title('Advanced Stock Chart')

symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)', 'AAPL')
start_date = st.sidebar.date_input('Select Start Date', datetime(2018, 1, 1))
end_date = st.sidebar.date_input('Select End Date', datetime.now())

indicators = st.sidebar.multiselect('Select Indicators', ['SMA', 'EMA','RSI','MACD'])
overlays = st.sidebar.multiselect('Select Overlays', ['Bollinger Bands'])
annotations = st.sidebar.text_input('Add Text Annotations (comma-separated)', 'This is an important event')
drawing_tool = st.sidebar.checkbox("Enable Trend Line Drawing Tool")

# Fetch data and plot chart
try:
    # Fetch stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Plot stock chart
    plot_stock_chart(stock_data, indicators, overlays, annotations.split(','), drawing_tool)
except Exception as e:
    st.error(f"Error fetching data: {e}")


