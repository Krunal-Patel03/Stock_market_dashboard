import streamlit as st
import pandas_datareader.data as web

# Function to fetch ticker symbols from a specific exchange
def fetch_ticker_symbols(exchange, num_symbols):
    if exchange.lower() == 'nasdaq':
        symbols = web.get_nasdaq_symbols()
    # Add more exchanges as needed
    return symbols.head(num_symbols)

# Streamlit app
def main():
    st.title("Ticker Symbol List")
    
    # Slider input to select the number of ticker symbols to display
    num_symbols = st.slider("Number of Ticker Symbols", min_value=1, max_value=100, value=10)
    
    # Dropdown to select exchange
    exchange = st.selectbox("Select Exchange", ["NASDAQ"])  # Add more exchanges as needed
    
    # Button to fetch ticker symbols
    if st.button("Fetch Ticker Symbols"):
        symbols = fetch_ticker_symbols(exchange, num_symbols)
        st.write("Ticker Symbols:")
        st.write(symbols)

if __name__ == "__main__":
    main()
