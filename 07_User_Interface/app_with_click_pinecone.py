
import streamlit as st
from streamlit_plotly_events import plotly_events
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta

# Model Definitions
pc = Pinecone(api_key="bc4ea65c-d63e-48e4-9b65-53d6272d927d")
embedding_model = SentenceTransformer("all-mpnet-base-v2")


# Function to fetch stock data
def fetch_data(ticker, start, end):
    """
    Fetch stock data for a given ticker within a specified date range.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date in 'YYYY-MM-DD' format.
    end (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: DataFrame containing stock data with date as index.
    """
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Function to plot stock data
def plot_data(stock_data, ticker):
    """
    Plot stock data for a given ticker.

    Parameters:
    stock_data (pandas.DataFrame): DataFrame containing stock data with date as index.
    ticker (str): The stock ticker symbol.

    Returns:
    plotly.graph_objs._figure.Figure: Plotly figure object with the stock price plot.
    """
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=ticker))
    #fig.update_layout(title=f'Stock Prices for {ticker}',
    #                  xaxis_title='Date',
    #                  yaxis_title='Stock Price (USD)')
    #fig, ax = plt.subplots()
    #fig.set_size_inches(18.5, 6, forward=True)
    #st.write(stock_data)
    fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'Stock Prices for {ticker}')
    fig.update_layout(title=f'Stock Prices for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Stock Price (USD)')
    fig.update_traces(hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y}')
    try:
        selected_points = plotly_events(fig)
        graph_date = selected_points[0]['x']
        date_object = datetime.strptime(graph_date, "%Y-%m-%d")
        #st.write(graph_date)
    except:
        date_object = datetime(2019, 1, 1)
    
    #tolerance = 2 # points
    #ax.plot(stock_data.index, stock_data['Close'], 'ro-', picker=tolerance)
    
    #def onclick(event):
    #    line = event.artist
    ##    xdata, ydata = line.get_data()
     #   ind = event.ind
     #   datetime_obj  = xdata[ind][0].astype('datetime64[s]').astype(datetime)
     #   formatted_date = datetime_obj.strftime('%Y-%m-%d')
     #   st.write(formatted_date)

    #cid = fig.canvas.mpl_connect('pick_event', onclick)
    return fig, date_object

# Function to find closest date before the selected date
def find_closest_date(sec_df, ticker, selected_date):
    """
    Find the closest date before the selected date for a given ticker in the SEC filings DataFrame.

    Parameters:
    sec_df (pandas.DataFrame): DataFrame containing SEC filings data with 'Date' and 'Ticker' columns.
    ticker (str): The stock ticker symbol.
    selected_date (str): The selected date in 'YYYY-MM-DD' format.

    Returns:
    pandas.Series: Row from the DataFrame corresponding to the closest date before the selected date.
    """
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
    """
    Generate a summary using a language model based on a query and text.

    Parameters:
    query (str): The query to be answered by the language model.
    text (str): The text to be summarized, typically from an SEC filing.
    model: The language model used to generate the summary.

    Returns:
    str: The generated summary response from the language model.
    """
    query_context = f"This text is from an SEC Filing:\n\n{text}\n\n Please answer the following"
    total_query = query_context + query
    response = model.generate_content(total_query)
    return response.text

def ask_vector_query(query, top_results, ticker, date, pinecone_index="fastvectors"):

    query_embeddings = embedding_model.encode(query)

    # Convert the string to a datetime object
    date_obj = date#datetime.strptime(date, '%Y-%m-%d')

    # Generate the list of dates for the previous week
    previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 8)]
    print(previous_week_dates)
    # date filter for vector database
    filter={
            "ticker": {"$eq": ticker},
            "date": {"$in": previous_week_dates}
    }

    index = pc.Index(pinecone_index)

    output = index.query(
        namespace="ns1",
        vector=[float(i) for i in list(query_embeddings)],
        filter=filter,
        top_k=top_results,
        include_values=False,
        include_metadata=True
    )
    print(output['matches'])

    retrieved_text = ' '
    for i in range(len(output['matches'])):
        Chunk_UID = output['matches'][i]['id']
        chunk_text = df_chunk[df_chunk['Chunk_ID']==Chunk_UID]['Text Chunks'].values[0]
        print('#########################################')
        print(chunk_text)
        retrieved_text += ' ' + chunk_text

     # Prompt and response function

    query_context = f" \
        Given the text from a financial news article, analyze the content and produce a bulleted response to the provided query: \
        \
        **Constraints:** ONLY RESPOND USING THE PROVIDED Context \
        \
        The context from the financial news article excerpts below: \
        {retrieved_text} \
        \
        Query: \
        {query} \
        \
        Example Response: \
        - The stock went up because of X \
        - The stock price went down because of Y \
        - The stock was impacted by Z \
        "
        
    print('#####################################################################################')
    print('##############################    RESPONSE   ########################################')
    print('#####################################################################################')

    response = model.generate_content(query_context)
    return response.text


# Streamlit app layout
st.title('Stock Price Analyzer')

# Load CSV into a DataFrame
csv_path = os.path.join(os.pardir, '02_Cleaned_Data', 'SEC_Filings.csv')
sec_df = pd.read_csv(csv_path)

# Load Vector Chunk References
df1_chunk = pd.read_csv(os.path.join(os.pardir, '05_Create_Vector_DB', 'Article_Chunk_References_pt1.csv'))
df2_chunk = pd.read_csv(os.path.join(os.pardir, '05_Create_Vector_DB', 'Article_Chunk_References_pt2.csv'))
df3_chunk = pd.read_csv(os.path.join(os.pardir, '05_Create_Vector_DB', 'Article_Chunk_References_pt3.csv'))
df_chunk = pd.concat([df1_chunk, df2_chunk, df3_chunk], ignore_index=True)
#df_chunk


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

# Fixed dates for the plot
start_date = datetime(2019, 1, 1)
end_date = datetime.today()

# Fetch and plot stock data if a plot is shown and a ticker is selected
if st.session_state.plot_shown and st.session_state.selected_ticker:
    data = fetch_data(st.session_state.selected_ticker, start_date, end_date)
    if not data.empty:
        fig, date_object = plot_data(data, st.session_state.selected_ticker)
        #st.plotly_chart(fig)
        
        ####
        #selected_points = []
        #if len(selected_points) > 0:
        #    selected_points = plotly_events(fig)
        #    graph_date=selected_points[0]['x']
        #    st.write('tree')#st.write(graph_date)
        #else:
        #    st.write('Click Date on Chart To Proceed')
        ####
        
        # Add a date selector for the article
        selected_date = st.date_input('Choose date from graph or select a date within the range', date_object, key='selected_date')

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

# LLM Model setup.  (API Key needs to be in your environment variables)
key = 'GOOGLE_API_KEY'
GOOGLE_API_KEY = os.getenv(key)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Query input and response
if st.session_state.article_text:
    query = st.text_input('Enter your query:')
    if st.button('Generate Summary') and query:
        response = generate_summary(query, st.session_state.article_text, model)
        st.session_state.response = response
        #st.write('Response:')
        #st.write(response)
    try:
        st.write('Response:')
        st.write(st.session_state.response)
    except:
        pass
        
# Query input and response
#if st.session_state.article_text:
    #query = st.text_input('Enter your query:')
    if st.button('Generate Vector Summary') and query:
        response_vector = ask_vector_query(query, 3, ticker, selected_date, pinecone_index="fastvectors")
        st.session_state.response_vector = response_vector
        #st.write('Response:')
        #st.write(response_vector)
    try:
        st.write('Response:')
        st.write(st.session_state.response_vector)
    except:
        pass
