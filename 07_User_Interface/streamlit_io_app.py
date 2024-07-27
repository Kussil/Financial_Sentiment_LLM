import streamlit as st
from streamlit_plotly_events import plotly_events
import yfinance as yf
import plotly.express as px
import pandas as pd
import os
from datetime import datetime, timedelta
import platform
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Model Definitions
pc = Pinecone(api_key="bc4ea65c-d63e-48e4-9b65-53d6272d927d")
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Function to fetch stock data
@st.cache_data
def fetch_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data['Change'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna()
    return stock_data

# Function to plot stock data
def plot_data(stock_data, ticker):
    min_date = stock_data.index.min().to_pydatetime()
    max_date = stock_data.index.max().to_pydatetime()
    date_range = st.slider("Select Graph Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    filtered_data = stock_data.loc[date_range[0]:date_range[1]]

    fig = px.line(filtered_data, x=filtered_data.index, y='Close', title=f'Stock Prices for {ticker}')
    fig.update_layout(title=f'Stock Prices for {ticker}', xaxis_title='Date', yaxis_title='Stock Price (USD)')
    fig.update_traces(hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y}<br>Change: %{customdata:.2f}%')
    fig.for_each_trace(lambda trace: trace.update(customdata=filtered_data['Change'].values * 100))

    try:
        selected_points = plotly_events(fig)
        graph_date = selected_points[0]['x']
        date_object = datetime.strptime(graph_date, "%Y-%m-%d")
        st.session_state.stock_change = filtered_data['Change'].loc[graph_date]
    except Exception:
        date_object = max_date
        st.session_state.stock_change = filtered_data['Change'].loc[graph_date]

    st.session_state.date_object = date_object
    return fig, date_object

# Function to plot sentiment
def plot_sentiment(sentiment_data, ticker, date_obj, num_days_back):
    if platform.system() == "Windows":
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%#m/%#d/%Y') for i in range(0, num_days_back + 1)]
    else:
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%-m/%-d/%Y') for i in range(0, num_days_back + 1)]

    sent_data = sentiment_data[(sentiment_data['Ticker'] == ticker)]
    sent_data = sent_data[sent_data['Date'].isin(previous_week_dates)]

    columns = ["Finance", "Production", 'Reserves / Exploration / Acquisitions / Mergers / Divestments',
               'Environment / Regulatory / Geopolitics', 'Alternative Energy / Lower Carbon',
               'Oil Price / Natural Gas Price / Gasoline Price']

    summary = sent_data[columns].apply(pd.Series.value_counts).fillna(0).T.reset_index()
    summary_melted = pd.melt(summary, id_vars=['index'], var_name='Sentiment', value_name='Article Count')

    fig = px.bar(summary_melted, x='Article Count', y='index', color='Sentiment', orientation='h',
                 labels={'index': 'Category', 'Article Count': 'Article Count', 'Sentiment': 'Sentiment'},
                 title='Sentiment Counts', barmode='stack',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'})

    fig.update_layout(title={'text': 'Sentiment Counts', 'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'})
    fig.update_xaxes(tickmode='array', tickvals=list(range(0, len(summary))))

    return fig

# Function to ask vector query
def ask_vector_query(query, top_results, ticker, date, pinecone_index, num_days_back, df_chunk, up_down):
    query_embeddings = embedding_model.encode(query)
    if date is None:
        filter = {"ticker": {"$eq": ticker}}
    elif date is not None and up_down is not None:
        up_down_indicator = 1 if up_down == 'increase' else 0
        date_obj = date
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, num_days_back + 1)]
        filter = {"ticker": {"$eq": ticker}, "date": {"$in": previous_week_dates}, "up_down_prediction": {"$eq": str(up_down_indicator)}}
    else:
        date_obj = date
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, num_days_back + 1)]
        filter = {"ticker": {"$eq": ticker}, "date": {"$in": previous_week_dates}}

    index = pc.Index(pinecone_index)
    output = index.query(namespace="ns1", vector=[float(i) for i in list(query_embeddings)], filter=filter,
                         top_k=top_results, include_values=False, include_metadata=True)

    article_ids = []
    retrieved_text = ' '
    for match in output['matches']:
        Chunk_UID = match['id']
        chunk_text = df_chunk[df_chunk['Chunk_ID'] == Chunk_UID]['Text Chunks'].values[0]
        article_ids.append('-'.join(Chunk_UID.split('-')[:2]))
        retrieved_text += ' ' + chunk_text

    query_context = f"Given the text from a financial news article, analyze the content and produce a bulleted response to the provided query: \
    **Constraints:** ONLY RESPOND USING THE PROVIDED Context \
    The context from the financial news article excerpts below: \
    {retrieved_text} \
    Query: \
    {query} \
    Example Response: \
    - **subject**: explanation \
    - **subject**: explanation \
    - **subject**: explanation"

    response = model.generate_content(query_context)
    return response.text, article_ids

# Streamlit app layout
st.markdown("<h1 style='text-align: center;'>FAST OG: Stock Price Analyzer</h1>", unsafe_allow_html=True)

# Define the base directory relative to the current script location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Helper function to load CSV files
@st.cache_data
def load_csv(file_path):
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        print(f"File does not exist: {full_path}")
        return pd.DataFrame()

# Load sentiment results
df_sentiment = load_csv('03_Sentiment_Analysis/Gemini/Prompt2/Prompt2_Sentiment_Analysis_Results.csv')

# Load Vector Full Article References
full_files = ['05_Create_Vector_DB/Gemini/Article_Full_References_pt1.csv', '05_Create_Vector_DB/Gemini/Article_Full_References_pt2.csv', '05_Create_Vector_DB/Gemini/Article_Full_References_pt3.csv']
df_full = pd.concat([load_csv(file) for file in full_files], ignore_index=True)
df_full['Unique_ID'] = df_full['Chunk_ID'].apply(lambda x: '-'.join(x.split('-')[:2]))

# Load Article Headline and URL References
columns_to_load = ['Source', 'Unique_ID', 'Date', 'Article Headline', 'URL']
article_files = ['02_Cleaned_Data/Investment_Research_Part1.csv', '02_Cleaned_Data/Investment_Research_Part2.csv',
                 '02_Cleaned_Data/ProQuest_Articles.csv', '02_Cleaned_Data/Earnings_Presentations.csv',
                 '02_Cleaned_Data/Earnings_QA.csv', '02_Cleaned_Data/SEC_Filings.csv']
articles_dfs = [load_csv(file)[columns_to_load] for file in article_files]
articles_df = pd.concat(articles_dfs, ignore_index=True)

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
if 'date_object' not in st.session_state:
    st.session_state.date_object = None
if 'response_vector' not in st.session_state:
    st.session_state.response_vector = None
if 'stock_change' not in st.session_state:
    st.session_state.stock_change = None
if 'articles_df_filtered' not in st.session_state:
    st.session_state.articles_df_filtered = None

# Fixed dates for the plot
start_date = pd.to_datetime(articles_df['Date']).min()
end_date = pd.to_datetime(articles_df['Date']).max() - pd.Timedelta(days=5)

# Fetch and plot stock data if a plot is shown and a ticker is selected
if st.session_state.plot_shown and st.session_state.selected_ticker:
    data = fetch_data(st.session_state.selected_ticker, start_date, end_date)
    if not data.empty:
        fig, date_object = plot_data(data, st.session_state.selected_ticker)

        # Add a date selector for the article
        selected_date = st.date_input('Choose date from graph or select a date within the range', date_object, key='selected_date')

        try:
            if st.session_state.stock_change > 0:
                up_down = 'increase'
            else:
                up_down = 'decrease'
            st.write(f'Daily stock price change on {selected_date}: **{abs(st.session_state.stock_change):.0%} {up_down}**')
        except Exception:
            pass

        st.divider()

        try:
            st.markdown("<h2 style='text-align: center;'>Sentiment Counts for Last Week of Articles</h2>", unsafe_allow_html=True)
            num_days_back = 7
            fig_sent = plot_sentiment(df_sentiment, st.session_state.selected_ticker, selected_date, num_days_back)
            st.plotly_chart(fig_sent)
        except Exception:
            pass

# LLM Model setup
key = 'AIzaSyC_hI1l9OTJhYoFw3UC-5LAfJXfENX9COs'
GOOGLE_API_KEY = os.getenv(key)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Query input and response
try:
    if selected_date is not None and st.session_state.stock_change is not None:
        st.markdown(f"<h2 style='text-align: center;'>What potential factors caused {ticker} stock price to {up_down} on {str(selected_date)}?</h2>", unsafe_allow_html=True)
        selected_chunk_count = 5
        query = f'What potential factors caused {ticker} stock price to {up_down}?'
        if st.button('Generate Response from Google Gemini') and query:
            response_vector, article_ids = ask_vector_query(query, selected_chunk_count, ticker, selected_date, "fastfullvectors", num_days_back, df_full, None)
            st.session_state.response_vector = response_vector.replace('$', '\$')
            print(st.session_state.response_vector)

        try:
            st.write('Response:')
            st.write(st.session_state.response_vector)
            try:
                st.session_state.articles_df_filtered = articles_df[articles_df['Unique_ID'].isin(article_ids)]
            except Exception:
                pass
            st.write('References:')
            for index, row in st.session_state.articles_df_filtered.iterrows():
                if st.checkbox(f"Show text for: Source: {row['Source']}, Article Headline: {row['Article Headline']}, Date: {row['Date']}, Article ID: {row['Unique_ID']}", key=row['Unique_ID']):
                    st.markdown(df_full[df_full['Unique_ID'] == row['Unique_ID']]['Text Chunks'].iloc[0])
        except Exception:
            pass
except Exception:
    pass

st.divider()

# Custom Generative AI Query
if st.session_state.response_vector:
    st.markdown("<h2 style='text-align: center;'>Custom Generative AI Query</h2>", unsafe_allow_html=True)
    ask_query = st.text_input('Ask your own query:')
    st.write(f'Toggle on to use full database, toggle off to use articles from {selected_date-timedelta(days=7)} to {selected_date}')
    on = st.checkbox("Use All Articles", value=True)
    ask_selected_date = None if on else selected_date
    selected_ask_chunk_count = 5
    if st.button('Ask Question') and ask_query:
        ask_response, ask_article_ids = ask_vector_query(ask_query, selected_ask_chunk_count, ticker, ask_selected_date, "fastfullvectors", num_days_back, df_full, None)
        st.session_state.ask_response = ask_response

    try:
        st.write('Response:')
        st.write(st.session_state.ask_response)
        st.write(articles_df[articles_df['Unique_ID'].isin(ask_article_ids)])
    except Exception:
        pass
