import subprocess
import talib as ta
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import plotly.graph_objs as go
from datetime import datetime
from colorama import Fore
from collections import defaultdict
import requests

st.set_page_config(page_title="Dashboard",layout="wide")
#Title 
st.title("StockExplorer")

#Ticker and date
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start-Date',datetime(2023, 1, 1))
end_date = st.sidebar.date_input('End-Date',datetime.now())

#the data
data = yf.download(ticker,start=start_date,end=end_date)
#Candlestick graph
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,
                                  open=data['Open'],
                                  high=data['High'],
                                  low=data['Low'],
                                  close=data['Close'], name='Candlesticks'))
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig)
with col2:
    col1, col2, col3 = st.columns(3)
    with col1:
        for i in range(18):
            st.write('<br>', unsafe_allow_html=True)
        if st.button("Advanced Chart"):
          st.switch_page("pages/advanced.py")
    with col2:
        company = yf.Ticker(ticker)
        st.subheader(company.info['shortName']+'('+ticker+')')
        st.write(f"<h7 style='text-align: right;'>Currency in {company.info['financialCurrency']}</h7>", unsafe_allow_html=True)
        st.write("<h4>Location:</h4>", unsafe_allow_html=True)
        st.write(company.info['state'],company.info['country'])
        st.write("<h4>Sector:</h4>", unsafe_allow_html=True)
        st.write(f"<h7 style='text-align: right;'>{company.info['sector']}</h7>", unsafe_allow_html=True)
        st.write("<h4>Industry:</h4>", unsafe_allow_html=True)
        st.write(company.info['industry'])
        st.write("<h4>Exchange & QuoteType:</h4>", unsafe_allow_html=True)
        st.write(company.info['exchange'],company.info['quoteType'])
    with col3:
        company = yf.Ticker(ticker)
        info = company.info    
        # Extracting required information
        stock_info = { 
            'Open': info.get('open', '-'),
            'High': info.get('dayHigh', '-'),
            'Low': info.get('dayLow', '-'),
            'Close': info.get('previousClose', '-'),
            'Dividend': info.get('dividendYield', '-'),
            'Beta': info.get('beta', '-'),
            'Volume': info.get('volume', '-'),
            'Avg.Vol': info.get('averageVolume', '-'),
            }           
         # Creating the table
        st.write(stock_info)
        # Extracting current day's range
        day_low = info.get('dayLow', '-')
        day_high = info.get('dayHigh', '-')           
        # Extracting 52-week range
        fifty_two_week_low = info.get('fiftyTwoWeekLow', '-')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', '-')
        # Slider for day range
        selected_day_range = st.slider("Day Range", float(day_low), float(day_high), (float(day_low), float(day_high)))            
        # Slider for 52-week range
        selected_fifty_two_week_range = st.slider("52-week Range", float(fifty_two_week_low), float(fifty_two_week_high), (float(fifty_two_week_low), float(fifty_two_week_high)))    

        

#For Fundamental Score
def get_scores(ticker):
    balance_sheet = ticker.balancesheet
    income_statement = ticker.income_stmt
    cash_flow = ticker.cashflow
    years = balance_sheet.columns

    pe_ratio = ticker.info.get('forwardPE', 0)

    # Profitability
    net_income = income_statement.loc['Net Income', years[0]]
    net_income_py = income_statement.loc['Net Income', years[1]]
    ni_score = 1 if net_income > 0 else 0
    ni_score_2 = 1 if net_income > net_income_py else 0

    op_cf = cash_flow.loc['Operating Cash Flow', years[0]]
    op_cf_score = 1 if op_cf > 0 else 0

    avg_assets = (balance_sheet.loc['Total Assets', years[0]] +
                  balance_sheet.loc['Total Assets', years[1]]) / 2
    avg_assets_py = (balance_sheet.loc['Total Assets', years[1]] +
                     balance_sheet.loc['Total Assets', years[2]]) / 2
    RoA = net_income / avg_assets
    RoA_py = net_income_py / avg_assets_py
    RoA_score = 1 if RoA > RoA_py else 0

    total_assets = balance_sheet.loc['Total Assets', years[0]]
    accruals = op_cf / total_assets - RoA
    ac_score = 1 if accruals > 0 else 0

    profitability_score = ni_score +  op_cf_score + RoA_score + ac_score

    # Leverage
    try:
        lt_debt = balance_sheet.loc['Long Term Debt', years[0]]
        lt_debt_py = balance_sheet.loc['Long Term Debt', years[1]]
        total_assets = balance_sheet.loc['Total Assets', years[0]]
        total_assets_py = balance_sheet.loc['Total Assets', years[1]]
        debt_ratio = lt_debt / total_assets
        debt_ratio_py = lt_debt_py / total_assets_py
        debt_ratio_score = 1 if debt_ratio < debt_ratio_py else 0
    except KeyError:
        debt_ratio_score = 1

    current_assets = balance_sheet.loc['Current Assets', years[0]]
    current_liab = balance_sheet.loc['Current Liabilities', years[0]]
    current_ratio = current_assets / current_liab
    current_assets_py = balance_sheet.loc['Current Assets', years[1]]
    current_liab_py = balance_sheet.loc['Current Liabilities', years[1]]
    current_ratio_py = current_assets_py / current_liab_py
    current_ratio_score = 1 if current_ratio > current_ratio_py else 0

    shares = balance_sheet.loc['Share Issued', years[0]]
    shares_py = balance_sheet.loc['Share Issued', years[1]]
    share_issued_score = 1 if shares_py == shares else 0

    leverage_score = debt_ratio_score + current_ratio_score + share_issued_score

    # Operating Efficiency
    gp = income_statement.loc['Gross Profit', years[0]]
    gp_py = income_statement.loc['Gross Profit', years[1]]
    revenue = income_statement.loc['Total Revenue', years[0]]
    revenue_py = income_statement.loc['Total Revenue', years[1]]
    gm = gp / revenue
    gm_py = gp_py / revenue_py
    gm_score = 1 if gm > gm_py else 0

    avg_assets = (balance_sheet.loc['Total Assets', years[0]] +
                  balance_sheet.loc['Total Assets', years[1]]) / 2
    avg_assets_py = (balance_sheet.loc['Total Assets', years[1]] +
                     balance_sheet.loc['Total Assets', years[2]]) / 2

    at = revenue / avg_assets
    at_py = revenue_py / avg_assets_py
    at_score = 1 if at > at_py else 0

    operating_efficiency_score = gm_score + at_score

    total_score = profitability_score + leverage_score + operating_efficiency_score

    return {
        'Profitability': profitability_score,
        'Leverage': leverage_score,
        'Operating Efficiency': operating_efficiency_score,
        'Total score': total_score
    }

company = yf.Ticker(ticker)
scores = get_scores(company)
company_info = company.info

# Fetch historical stock price data
stock_data = yf.download(ticker,start=datetime(2018,1,1),end=datetime.now())

# Calculate technical indicators using talib
roc = ta.ROC(stock_data['Close'], timeperiod=20)
rsi = ta.RSI(stock_data['Close'], timeperiod=14)
macd, macd_signal, _ = ta.MACD(stock_data['Close'])
adx = ta.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
slowk, slowd = ta.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
cci = ta.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=20)
mfi = ta.MFI(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'], timeperiod=14)
williams_r = ta.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
atr = ta.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
fastk, fastd = ta.STOCHRSI(stock_data['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

# Define indicator names and scores
indicators = [
    "RSI",
    "MACD",
    "ADX",
    "STOCH",
    "STOCHRSI",
    "CCI",
    "ATR",
    "ROC",
    "Williamson%R",
    "MFI"
]

# Define indicator values
values = [
    rsi[-1],
    macd[-1],
    adx[-1],
    slowk[-1],
    fastk[-1],
    cci[-1],
    atr[-1],
    roc[-1],
    williams_r[-1],
    mfi[-1],
]

Indication = []
def interpret_indicator(indicator_name, indicator_value):
    interpretation = ""

    # RSI
    if indicator_name == "RSI":
        if indicator_value < 30:
            interpretation =  "Oversold"
        elif indicator_value < 45:
            interpretation =  "Sell"
        elif 45 <= indicator_value < 55:
            interpretation =  "Neutral"
        elif 55 <= indicator_value < 75:
            interpretation =  "Buy"
        else:
            interpretation =  "Overbought"

    # MACD
    elif indicator_name == "MACD":
        if indicator_value > 0:
            interpretation =  "Buy"
        else:
            interpretation =  "Sell"

    # STOCH and STOCHRSI (Same logic)
    elif indicator_name in ["STOCH", "STOCHRSI","MFI"]:
        if indicator_value > 80:
            interpretation =  "Overbought"
        elif 55 <= indicator_value <= 80:
            interpretation =  "Buy"
        elif 45 <= indicator_value < 55:
            interpretation =  "Neutral"
        elif 20 <= indicator_value < 45:
            interpretation =  "Sell"
        else:
            interpretation =  "Oversold"

    # ROC
    elif indicator_name == "ROC":
        if indicator_value > 0:
            interpretation =  "Buy"
        elif indicator_value < 0:
            interpretation =  "Sell"
        else:
            interpretation =  "Neutral"

    # CCI
    elif indicator_name == "CCI":
        if indicator_value > 200:
            interpretation =  "Overbought"
        elif -200 <= indicator_value < -50:
            interpretation =  "Sell"
        elif -50 <= indicator_value < 50:
            interpretation =  "Neutral"
        elif 50 <= indicator_value < 200:
            interpretation =  "Buy"
        else:
            interpretation =  "Oversold"

    # Williamson %R
    elif indicator_name == "Williamson%R":
        if indicator_value <= -80:
            interpretation =  "Oversold"
        elif -80 < indicator_value <= -50:
            interpretation =  "Sell"
        elif -50 < indicator_value <= -20:
            interpretation =  "Buy"
        elif -20 < indicator_value <= 0:
            interpretation =  "Overbought"

   

    # ATR
    elif indicator_name == "ATR":
        sma9 = ta.SMA(stock_data['Close'], timeperiod=9)
        if indicator_value > sma9[-1]:
            interpretation =  "Highly Volatile"
        else:
            interpretation =  "Less Volatile"

    # ADX
    elif indicator_name == "ADX":
        if indicator_value < 25:
            interpretation =  "Weak Trend"
        elif 25 < indicator_value < 50:
            interpretation =  "Strong Trend"
        elif 50 < indicator_value < 75:
            interpretation =  "Very Strong Trend"
        else:
            interpretation =  "Extremely Strong Trend"

    return interpretation

for i,j in zip(indicators,values):
    res = interpret_indicator(i,j)
    Indication.append(res)

def Tech_calculate_score(indication):
    if indication in ["Buy", "Very Strong Trend", "Extremely Strong Trend", "Less Volatile"]:
        return 1
    elif indication in ["Sell", "Weak Trend", "Highly Volatile"]:
        return 0
    elif indication == "Neutral":
        return 0.5
    elif indication == "Strong Trend":
        return 0.75
    elif "Overbought" in indication:  # assuming "Overbought" can be part of other indications
        return 0.6
    else:
        return 0.4

def build_Tech_score(indications):
    total_score = 0
    for indication in indications:
        total_score += Tech_calculate_score(indication)
    return total_score

# Function to calculate color based on score
def get_color(score, max_score):
    if score >= max_score * 0.7:
        return "green"
    elif score >= max_score * 0.4:
        return "yellow"
    else:
        return "red"

# Function to display circular gauge
def display_circular_gauge(score, max_score):
    # Calculate progress percentage
    progress_percentage = (score / max_score) * 100

    # Create gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge=dict(
            axis=dict(range=[None, max_score], tickvals=[0, max_score], ticks=''),
            bar=dict(color=get_color(score, max_score)),
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, max_score], color="white")
            ],
            threshold=dict(
                line=dict(color="red", width=2),
                thickness=0.75,
                value=score
            )
        ),
        number={'suffix': f'/{max_score}', 'font': {'size': 20}}
    ))
    fig.update_layout(width=250, height=250)
    # Display the gauge
    return fig

#Sentimental Score
ticker = ticker
url = f'https://eodhd.com/api/sentiments?s={ticker}&from=2024-01-01&to=2024-05-15&api_token=662a3477190930.15607695&fmt=json'
s_data = requests.get(url).json()
s_key = list(s_data.keys())
comp_state = s_key[0]

# Extracting date and normalized values
dates = [entry['date'] for entry in s_data[comp_state]]
sentiments = [entry['normalized'] for entry in s_data[comp_state]]
# Initialize a dictionary to store the sum and count of sentiment scores for each month
monthly_sentiment_sum = defaultdict(float)
monthly_sentiment_count = defaultdict(int)

# Iterate through each data entry
for entry in s_data[comp_state]:
    # Extract year and month from the date
    year_month = entry['date'][:7]
    # Add the normalized sentiment score to the sum for the corresponding month
    monthly_sentiment_sum[year_month] += entry['normalized']
    # Increment the count of sentiment scores for the corresponding month
    monthly_sentiment_count[year_month] += 1

# Initialize a dictionary to store the average sentiment score for each month
monthly_average_sentiment = {}

# Calculate the average sentiment score for each month
for year_month in monthly_sentiment_sum:
    # Calculate the average sentiment score by dividing the sum by the count
    average_sentiment = monthly_sentiment_sum[year_month] / monthly_sentiment_count[year_month]
    # Store the average sentiment score for the month
    monthly_average_sentiment[year_month] = average_sentiment

# Extracting the latest sentiment score
latest_month = list(monthly_average_sentiment.keys())[0]
latest_score = round(monthly_average_sentiment[latest_month] * 10, 1)


st.write('<hr>', unsafe_allow_html=True)
#Price Prediction
# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.65)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

# Function to create sequences for LSTM input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 100  # Length of input sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape the input data to match the expected shape for LSTM
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=64)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Take the last sequence from the test data to start forecasting
last_sequence = X_test[-1]
future_steps = 30
# Make predictions for future time steps
forecast = []
for _ in range(future_steps):
    # Reshape the last sequence to match the input shape
    last_sequence_reshaped = last_sequence.reshape(1, seq_length, 1)
    # Predict the next value
    next_prediction = model.predict(last_sequence_reshaped)
    # Append the prediction to the forecast list
    forecast.append(next_prediction[0][0])  # Extracting the scalar value from the prediction
    # Update the last sequence by removing the first element and adding the prediction
    last_sequence = np.append(last_sequence[1:], next_prediction)

# Inverse transform the forecasted values to get the actual price
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Prepare data for plotting
dates = stock_data.index[-100:].strftime('%Y-%m-%d')
close_prices = stock_data['Close'].values[-100:]
forecast_dates = pd.date_range(start=stock_data.index[-1], periods=future_steps + 1).strftime('%Y-%m-%d')
forecast_prices = np.concatenate((stock_data['Close'].values[-1:], forecast.flatten()), axis=None)
next_day = forecast[0]
up_down_indication = "UP" if next_day > stock_data['Close'].iloc[-1] else "DOWN"
week_fore = forecast[:7]
# Create interactive plot
fig = go.Figure()
# Plot last 100 days close price
fig.add_trace(go.Scatter(x=dates, y=close_prices, mode='lines', name='Historical Close Price'))
# Plot forecasted prices
fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines', name='Forecast Price'))
# Update layout for better readability
fig.update_layout(title=f"Stock Price Forecast for {ticker}",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  xaxis_rangeslider_visible=True)
# Display the plot using Streamlit
col1, col2 = st.columns(2)
with col1:
    st.subheader("Price Forecasting")
    # Display the plot using Streamlit
    st.plotly_chart(fig)
with col2:
    st.subheader("StockExplorer Ratings")
    col1, col2,col3 = st.columns(3)
    with col1:
        st.write("<h5>Fundamental Score</h5>", unsafe_allow_html=True)
        st.write(display_circular_gauge(scores['Total score'],9))
    with col2:
        st.write("<h5>Technical Score</h5>", unsafe_allow_html=True)
        st.write(display_circular_gauge(build_Tech_score(Indication),10))
    with col3:
        st.write("<h5>Sentimental Score</h5>", unsafe_allow_html=True)
        st.write(display_circular_gauge(latest_score,10))
    st.write("<h5>Next Day Forecasting</h5>", unsafe_allow_html=True)
    st.write(f'{up_down_indication}(Forecasted Price:{next_day})')

stock_data = data
st.write('<hr>', unsafe_allow_html=True)
st.subheader("Techical Analysis")
#RSI AND Plotying RSI
data['RSI'] = ta.RSI(data['Close'])

# Create Plotly figure for interactive plot
fig1 = go.Figure()

# Add RSI trace
fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='blue')))

# Add horizontal lines at RSI levels
fig1.add_shape(type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70, line=dict(color="red", width=1, dash="dash"))
fig1.add_shape(type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30, line=dict(color="green", width=1, dash="dash"))

# Update layout
fig1.update_layout(
    title='Relative Strength Index (RSI)',
    xaxis_title='Date',
    yaxis_title='RSI',
    xaxis=dict(tickangle=45),
    hovermode="x unified"
)

# Display the plot using Streamlit
#st.plotly_chart(fig)

# Calculate MACD
macd, macd_signal, macd_hist = ta.MACD(stock_data['Close'])

# Create Plotly figure for interactive plot
fig = go.Figure()

# Add Close price trace
#fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))

# Add MACD and Signal traces
fig.add_trace(go.Scatter(x=stock_data.index, y=macd, mode='lines', name='MACD'))
fig.add_trace(go.Scatter(x=stock_data.index, y=macd_signal, mode='lines', name='Signal'))

# Determine color for MACD histogram bars
colors = ['red' if cl < 0 else 'green' for cl in macd_hist]

# Add MACD Histogram bars
fig.add_trace(go.Bar(x=stock_data.index, y=macd_hist, marker=dict(color=colors), name='MACD Histogram'))

# Update layout
fig.update_layout(
    title='MACD Analysis',
    xaxis_title='Date',
    yaxis_title='Value',
    xaxis=dict(tickangle=45),
    hovermode="x unified"
)
# Display the plots side by side using Streamlit
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1)
with col2:
    st.plotly_chart(fig)



# Create DataFrame for the technical analysis report
report_df = pd.DataFrame({
    'Indicator': indicators,
    'Value': values,
    'Indication' : Indication
})

# Define a function to apply the style to each row
def apply_style(row):
    if row['Indication'] in ['Buy','Strong Trend', 'Very Strong Trend', 'Extremely Strong Trend']:
        return ['background-color: green'] * len(row)
    elif row['Indication'] in ['Sell','Weak Trend']:
        return ['background-color: red'] * len(row)
    else:
        return ['background-color: gray'] * len(row)

# Display the technical analysis report using Streamlit
st.write("<h5 style='text-align: center;'>Technical Analysis Report</h5>", unsafe_allow_html=True)
report_df = report_df.style.apply(apply_style, axis=1)
st.table(report_df)
# Apply styling and alignment to the table
#st.table(report_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
#    'selector': 'th',
#    'props': [('text-align', 'center'), ('font-size', '16px')]
#}, {
#    'selector': 'td',
#    'props': [('text-align', 'center'), ('font-size', '14px')]
#}]))

# Display the plot using Streamlit
#st.plotly_chart(fig)

#plotting sma100 and ema100
#Moving Averages
data['SMA_100'] = ta.SMA(stock_data['Close'],100)
data['SMA_20'] = ta.SMA(data['Close'],20)
data['EMA_100'] = ta.EMA(stock_data['Close'],100)
data['EMA_20'] = ta.EMA(data['Close'],20)
# Create Plotly figure for SMA_100
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=stock_data.index, y=data['SMA_20'], mode='lines', name='SMA_20', line=dict(color='orange')))
fig1.update_layout(title='SMA_20', xaxis_title='Date', yaxis_title='Price')

# Create Plotly figure for EMA_100
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=stock_data.index, y=data['EMA_20'], mode='lines', name='EMA_20', line=dict(color='green')))
fig2.update_layout(title='EMA_20', xaxis_title='Date', yaxis_title='Price')


# Calculate moving averages for different time periods
moving_average_periods = [5, 9, 10, 20, 30, 50, 100, 200]
moving_average_data = {}
for period in moving_average_periods:
    sma = ta.SMA(stock_data['Close'], timeperiod=period)
    ema = ta.EMA(stock_data['Close'], timeperiod=period)
    moving_average_data[f'SMA_{period}'] = sma.iloc[-1]  # Selecting the last value
    moving_average_data[f'EMA_{period}'] = ema.iloc[-1]  # Selecting the last value

# Create DataFrame for moving average values
sma_indication = []
ema_indication = []
last_close_price = stock_data['Close'].iloc[-1]
for period in moving_average_periods:
    sma_ind = 'Buy' if last_close_price > moving_average_data[f'SMA_{period}'] else 'Sell'
    sma_indication.append(sma_ind)
for period in moving_average_periods:
    ema_ind = 'Buy' if last_close_price > moving_average_data[f'EMA_{period}'] else 'Sell'
    ema_indication.append(ema_ind)

df_data = {
    'Moving Average': ['MA5','MA9', 'MA10', 'MA20', 'MA30','MA50', 'MA100', 'MA200'],
    'SMA': [moving_average_data['SMA_5'],moving_average_data['SMA_9'],moving_average_data['SMA_10'],
            moving_average_data['SMA_20'],moving_average_data['SMA_30'] ,moving_average_data['SMA_50'], 
            moving_average_data['SMA_100'], moving_average_data['SMA_200']],
    'Indication' : sma_indication,
    'EMA': [moving_average_data['EMA_5'],moving_average_data['EMA_9'],moving_average_data['EMA_10'],
             moving_average_data['EMA_20'],moving_average_data['EMA_30'],moving_average_data['EMA_50'], 
            moving_average_data['EMA_100'], moving_average_data['EMA_200']],
    'Signal' : ema_indication,
}
df = pd.DataFrame(df_data)

signal = []
short = 'Buy' if moving_average_data["SMA_5"] > moving_average_data['SMA_20'] else 'Sell'
medium = 'Buy' if moving_average_data["SMA_20"] > moving_average_data['SMA_50'] else 'Sell'
long = 'Buy' if moving_average_data["SMA_50"] > moving_average_data['SMA_200'] else 'Sell'
signal.append(short)
signal.append(medium)
signal.append(long)
mix_ma = {
    "Period" : ["Short Term","Medium Term","Long Term"],
    "Crossover" : ["5&20 Crossover","20&50 Crossover","50&200 Crossover"],
    "Indication" : signal
}
ma_score = 0
ma_buy = sma_indication.count("Buy")+ema_indication.count("Buy")+signal.count("Buy")
ma_sell = sma_indication.count("Sell")+ema_indication.count("Sell")+signal.count("Sell")
ma_score = ma_buy - ma_sell
if ma_buy>ma_sell:
    ma_score += 1
ma_score = ma_score/2
# Display the plots side by side using Streamlit
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1)
with col2:
    st.plotly_chart(fig2)

mix_ma = pd.DataFrame(mix_ma)
mix_ma = mix_ma.style.apply(apply_style, axis=1)

col1, col3 = st.columns(2)
with col1:
    # Apply CSS styling to center the table vertically
    st.write("<h6 style='text-align: center;'>Moving Averages Table</h6>", unsafe_allow_html=True)
    st.table(df.style.set_table_attributes(f'style="font-size: 20px; margin-top: 50px;"'))
    col1,col2 = st.columns(2)
    with col1:
        st.write("<h6 style='text-align: center;'>Moving Averages Crossover Table</h6>", unsafe_allow_html=True)
        st.table(mix_ma)
    with col2:
        st.write("<h5 style='text-align: left;'>SMA Indication:</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: right;'>Buy:{sma_indication.count("Buy")}</h6>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: right;'>Sell:{sma_indication.count("Sell")}</h6>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: left;'>EMA Indication:</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: right;'>Buy:{ema_indication.count("Buy")}</h6>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: right;'>Sell:{ema_indication.count("Sell")}</h6>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: left;'>Crossover Indication:</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: right;'>Buy:{signal.count("Buy")}</h6>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: right;'>Sell:{signal.count("Sell")}</h6>", unsafe_allow_html=True)

with col3:
        num2,num3 = st.columns(2)
        with num2:
            for i in range(16):
                st.write('<br>', unsafe_allow_html=True)
            st.write("<h5>Moving Average Score</h5>", unsafe_allow_html=True)
            st.write(display_circular_gauge(ma_score,10))
        with num3:
            for i in range(16):
                st.write('<br>', unsafe_allow_html=True)
            st.write("<h5>Technical Indicator Score</h5>", unsafe_allow_html=True)
            st.write(display_circular_gauge(build_Tech_score(Indication),10))

st.write('<hr>', unsafe_allow_html=True)
st.subheader("Fundamental Analysis")


# Function to display score with color and progress bar
def display_score_with_color_and_progress_bar(label, score, max_score):
    st.write(f"<h6>{label}</h6>", unsafe_allow_html=True)
    progress_color = get_color(score, max_score)
    progress_percentage = (score / max_score)  # Adjusted to be within [0.0, 1.0]
    st.progress(progress_percentage)
    st.write(f"<h5 style='color: {progress_color}; text-align: right;'>{score}/{max_score}</h5>", unsafe_allow_html=True)


col1,col2,col3,col4 = st.columns(4)
with col1:
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h3>StockExplorer Score</h3>", unsafe_allow_html=True)
    st.write(display_circular_gauge(scores['Total score'],9))
    st.write('<hr>', unsafe_allow_html=True)
    display_score_with_color_and_progress_bar("Profitability Score", scores['Profitability'], 4)
    display_score_with_color_and_progress_bar("Operating Efficiency", scores['Operating Efficiency'], 2)
    display_score_with_color_and_progress_bar("Leverage Score", scores['Leverage'], 3)
    st.write('<hr>', unsafe_allow_html=True)

with col2:
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h3>Company's</h3>", unsafe_allow_html=True)
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h4>Market Cap</h4>", unsafe_allow_html=True)
    st.write(f"<h5>${company_info.get("marketCap", 0)}</h5>", unsafe_allow_html=True)
    st.write("<h4>Current Price</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("currentPrice", 0)}</h5>", unsafe_allow_html=True)
    st.write("<h4>High / Low:</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("dayHigh", 0)}/${company_info.get("dayLow", 0)}</h5>", unsafe_allow_html=True)
    st.write("<h4>Book Value</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("bookValue", 0)}</h5>", unsafe_allow_html=True)
    st.write("<h4>Total Cash</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("totalCash", 0)}</h5>", unsafe_allow_html=True)
    st.write("<h4>Number Of Shares:</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("sharesOutstanding", 0)}</h5>", unsafe_allow_html=True)
    st.write('<hr>', unsafe_allow_html=True)

with col3:
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h3>Essential</h3>", unsafe_allow_html=True)
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h4>ROCE</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{company_info.get("returnOnCapitalEmployed", 0)}%</h5>", unsafe_allow_html=True)
    st.write("<h4>ROE</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{company_info.get("returnOnEquity", 0)*100}%</h5>", unsafe_allow_html=True)
    st.write("<h4>Dividend Yield</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{company_info.get("dividendYield", 0)*100}%</h5>", unsafe_allow_html=True)
    st.write("<h4>Promoter Holding</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{(company_info.get("heldPercentInsiders", 0))*100}%</h5>", unsafe_allow_html=True)
    st.write("<h4>Profit Margin</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{(company_info.get('profitMargins', 0))*100}%</h5>", unsafe_allow_html=True)
    st.write("<h4>Revenue Growth</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{(company_info.get('revenueGrowth', 0))*100}%</h5>", unsafe_allow_html=True)
    st.write('<hr>', unsafe_allow_html=True)

with col4:
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h3>Fundamentals</h3>", unsafe_allow_html=True)
    st.write('<hr>', unsafe_allow_html=True)
    st.write("<h4>P/E</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{company_info.get("trailingPE", "N/A")}</h5>", unsafe_allow_html=True)
    st.write("<h4>P/B Ratio</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{company_info.get("priceToBook", "N/A")}</h5>", unsafe_allow_html=True)
    st.write("<h4>Face Value</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("faceValue", 0)}</h5>", unsafe_allow_html=True)
    st.write("<h4>EPS</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >{company_info.get("trailingEps", "N/A")}</h5>", unsafe_allow_html=True)
    st.write("<h4>Enterprise Value</h4>", unsafe_allow_html=True)
    st.write(f"<h5 >${company_info.get("enterpriseValue", 0)}</h5>", unsafe_allow_html=True)
    st.write("You can add more!!!!")
    st.button("Add More")
    st.write('<hr>', unsafe_allow_html=True)

# Get income statement data
EPS = company.financials.loc['Basic EPS']
revenue = company.financials.loc['Total Revenue']
net_income = company.financials.loc['Net Income']
earning = company.financials.loc['EBITDA']

# Plotting Earnings as line plot
fig_earning = go.Figure()
fig_earning.add_trace(go.Scatter(x=earning.index, y=earning, name='Earnings'))
fig_earning.update_layout(title=f"{ticker} Earnings",
                          xaxis_title="Date",
                          yaxis_title="Earnings",
                          hovermode='x unified')

# Plotting EPS as line plot
fig_EPS = go.Figure()
fig_EPS.add_trace(go.Scatter(x=EPS.index, y=EPS, name='EPS'))
fig_EPS.update_layout(title=f"{ticker} Basic EPS",
                      xaxis_title="Date",
                      yaxis_title="EPS",
                      hovermode='x unified')

# Creating bar chart for Revenue and Net Income
fig_revenue_net_income = go.Figure()
fig_revenue_net_income.add_trace(go.Bar(x=revenue.index, y=revenue, name='Revenue'))
fig_revenue_net_income.add_trace(go.Bar(x=net_income.index, y=net_income, name='Net Income'))
fig_revenue_net_income.update_layout(title=f"{ticker} Revenue and Net Income",
                                     xaxis_title="Date",
                                     yaxis_title="Amount",
                                     hovermode='x unified',
                                     barmode='group')

col1,col2 = st.columns(2)
with col1:
    tab1,tab2 = st.tabs(["Earnings","EPS"])
    with tab1:
        st.plotly_chart(fig_earning)
    with tab2:
         st.plotly_chart(fig_EPS)
with col2:
    st.plotly_chart(fig_revenue_net_income)

# Function to fetch financial data
def fetch_financial_data(symbol):
    company = yf.Ticker(symbol)
    quarterly_results = company.quarterly_incomestmt
    balance_sheet = company.balance_sheet
    profit_loss = company.financials
    cash_flow = company.cashflow
    return quarterly_results, balance_sheet, profit_loss, cash_flow

# Define the ticker symbol
ticker_symbol = ticker

# Fetch financial data
quarterly_results, balance_sheet, profit_loss, cash_flow = fetch_financial_data(ticker_symbol)

# Expander for quarterly results
with st.expander("Quarterly Results"):
    st.write(quarterly_results)

# Expander for balance sheet
with st.expander("Balance Sheet"):
    st.write(balance_sheet)

# Expander for profit & loss
with st.expander("Profit & Loss"):
    st.write(profit_loss)

# Expander for cash flow statement
with st.expander("Cash Flow Statement"):
    st.write(cash_flow)

st.write('<hr>', unsafe_allow_html=True)
st.subheader("Setimental Analysis")

# Create Plotly figure
fig = go.Figure()
# Add trace
fig.add_trace(go.Scatter(x=dates, y=sentiments, mode='lines', name='Sentiment'))
# Update layout
fig.update_layout(title=f'Sentiment Analysis for {comp_state}',
                  xaxis_title='Date',
                  yaxis_title='Sentiment',
                  xaxis=dict(tickangle=45),
                  template='plotly_dark')

# Creating a list of rounded sentiment scores
sent_score = [round(score * 10, 1) for score in monthly_average_sentiment.values()]
# Creating a table of monthly sentiment scores
sent_table = {
    "Month": list(monthly_average_sentiment.keys()),
    "Score": sent_score
}

col1,col2 = st.columns(2)

with col1:
    st.write("<h5 style='text-align: left'>StockExplorer Sentiment Score</h5>", unsafe_allow_html=True)
    st.write(display_circular_gauge(latest_score,10))
    st.write("<h6 style='text-align: center;'>Monthly Sentimental Score</h6>", unsafe_allow_html=True)
    st.table(sent_table)
with col2:
    st.plotly_chart(fig)