import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import os
import google.generativeai as genai

# Function to fetch stock data
def fetch_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Function to plot stock data
def plot_data(stock_data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=ticker))
    fig.update_layout(title=f'Stock Prices for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Stock Price (USD)')
    return fig

# Function to find closest date before the selected date
def find_closest_date(sec_df, ticker, selected_date):
    sec_df['Date'] = pd.to_datetime(sec_df['Date'])
    selected_datetime = pd.to_datetime(selected_date)
    ticker_data = sec_df[sec_df['Ticker'] == ticker]
    ticker_data = ticker_data[ticker_data['Date'] <= selected_datetime]
    if not ticker_data.empty:
        closest_date_row = ticker_data.loc[ticker_data['Date'].idxmax()]
        return closest_date_row
    return None

# Function to generate summary using LLM
def generate_summary(query, text, model):
    query_context = f"This text is from an SEC Filing:\n\n{text}\n\n Please answer the following"
    total_query = query_context + query
    response = model.generate_content(total_query)
    return response.text

# Streamlit app layout
st.title('Stock Price Analyzer')

# Load CSV into a DataFrame
csv_path = os.path.join(os.pardir, '02_Cleaned_Data', 'SEC_Filings.csv')
sec_df = pd.read_csv(csv_path)

# Default tickers
default_tickers = ['BP', 'COP', 'CVX', 'CXO', 'DVN', 'EOG', 'EQNR', 'HES', 'MPC', 'MRO', 'OXY', 'PDCE', 'PSX', 'PXD', 'SHEL', 'TTE', 'VLO', 'XOM']

# Dropdown for tickers with blank default and single selection
ticker = st.selectbox('Select a stock ticker', options=[''] + default_tickers)

# Use Streamlit session state to remember selected ticker and plot state
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None
if 'plot_shown' not in st.session_state:
    st.session_state.plot_shown = False
if 'article_text' not in st.session_state:
    st.session_state.article_text = ""

if st.button('Show Plot'):
    st.session_state.selected_ticker = ticker
    st.session_state.plot_shown = True

# Fixed dates
start_date = datetime(2019, 1, 1)
end_date = datetime.today()

if st.session_state.plot_shown and st.session_state.selected_ticker:
    data = fetch_data(st.session_state.selected_ticker, start_date, end_date)
    if not data.empty:
        fig = plot_data(data, st.session_state.selected_ticker)
        st.plotly_chart(fig)

        # Add a date selector for the article
        selected_date = st.date_input('Select a date within the range', start_date, key='selected_date')

        # When a date is selected, find the closest date and display the article text
        if selected_date:
            closest_date_row = find_closest_date(sec_df, st.session_state.selected_ticker, selected_date)
            if closest_date_row is not None:
                st.session_state.article_text = closest_date_row['Article Text']
                st.markdown(
                    f"**Article found:** "
                    f"{closest_date_row['Article Headline']} "
                    f"({closest_date_row['Date'].strftime('%Y-%m-%d')})"
                )
            else:
                st.write(f"No articles found before {selected_date}")
                st.session_state.article_text = ""
    else:
        st.write(f'No data found for {st.session_state.selected_ticker}')
        st.session_state.article_text = ""

# LLM Model setup
key = 'GOOGLE_API_KEY'
GOOGLE_API_KEY = os.getenv(key)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Query input and response
if st.session_state.article_text:
    query = st.text_input('Enter your query:')
    if st.button('Generate Summary') and query:
        response = generate_summary(query, st.session_state.article_text, model)
        st.write('Response:')
        st.write(response)
