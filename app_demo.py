
import streamlit as st
from streamlit_plotly_events import plotly_events
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
import platform

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
    stock_data['Change'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna()
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
    min_date = stock_data.index.min().to_pydatetime()
    max_date = stock_data.index.max().to_pydatetime()

    date_range = st.slider("Select Graph Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    filtered_data = stock_data.loc[date_range[0]:date_range[1]]

    fig = px.line(filtered_data, 
                  x=filtered_data.index, 
                  y='Close', 
                  title=f'Stock Prices for {ticker}')
    fig.update_layout(title=f'Stock Prices for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Stock Price (USD)')
    fig.update_traces(hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y}<br>Change: %{customdata:.2f}%', line_color='#0A509E')
    fig.for_each_trace(lambda trace: trace.update(customdata=filtered_data['Change'].values * 100))

    try:
        selected_points = plotly_events(fig)
        graph_date = selected_points[0]['x']
        date_object = datetime.strptime(graph_date, "%Y-%m-%d")
        if graph_date in filtered_data.index:
            st.session_state.stock_change = filtered_data['Change'].loc[graph_date]
        else:
            nearest_date = filtered_data.index.get_loc(graph_date, method='nearest')
            nearest_date_str = filtered_data.index[nearest_date].strftime("%Y-%m-%d")
            st.session_state.stock_change = filtered_data['Change'].loc[nearest_date_str]
            date_object = filtered_data.index[nearest_date]
    except Exception as e:
        #st.error(f"An error occurred: {e}")
        default_date = filtered_data.index[-1]  # Use the last date in filtered_data as the default
        date_object = default_date
        st.session_state.stock_change = filtered_data['Change'].loc[default_date]
    
    st.session_state.date_object = date_object
    return fig, date_object


def plot_sentiment(sentiment_data, ticker, date_obj, num_days_back):
    """
    Plots sentiment analysis results based on the given sentiment_data DataFrame,
    for a specific ticker and date range.

    Parameters:
    - sentiment_data (DataFrame): DataFrame containing sentiment analysis data
                                  with columns 'Ticker' and 'Date' among others.
    - ticker (str): Ticker symbol for which sentiment analysis is plotted.
    - date_obj (datetime.date): Reference date for plotting sentiment data.
    - num_days_back (int): Number of days back from date_obj to consider for plotting.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure object containing the sentiment
                                      analysis bar chart.
    """
    
    if platform.system() == "Windows":
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%#m/%#d/%Y') for i in range(0, num_days_back + 1)]
    else:
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%-m/%-d/%Y') for i in range(0, num_days_back + 1)]

    sent_data = sentiment_data[(sentiment_data['Ticker'] == ticker)]
    sent_data = sent_data[sent_data['Date'].isin(previous_week_dates)]
    
    columns = ["Finance", 
               "Production", 
               'Reserves / Exploration / Acquisitions / Mergers / Divestments', 
               'Environment / Regulatory / Geopolitics', 
               'Alternative Energy / Lower Carbon', 
               'Oil Price / Natural Gas Price / Gasoline Price']

    # Get the total count of positive, neutral, and negative for each column
    summary = sent_data[columns].apply(pd.Series.value_counts).fillna(0)
    # Transpose the summary DataFrame to have columns as the index
    summary = summary.T
    print(summary)

    # Reset index to turn index into regular columns for Plotly
    summary = summary.reset_index()

    # Melt DataFrame to have Sentiment as a categorical variable
    summary_melted = pd.melt(summary, id_vars=['index'], var_name='Sentiment', value_name='Article Count')
    
    # Plotting with Plotly Express
    fig = px.bar(summary_melted, x='Article Count', y='index', color='Sentiment', orientation='h',
                 labels={'index': 'Category', 'Article Count': 'Article Count', 'Sentiment': 'Sentiment'},
                 title='Sentiment Counts',
                 barmode='stack',
                 color_discrete_map = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'})
    
    fig.update_layout(title={'text': 'Sentiment Counts',
                         'x': 0.5,  # Center align title horizontally
                         'y': 0.95,  # Position title slightly above the plot
                         'xanchor': 'center',  # Center align title horizontally
                         'yanchor': 'top'})  # Position title slightly above the plot
    
    # Update the x-axis to show integer ticks only
    fig.update_xaxes(tickmode='array', tickvals=list(range(0, len(sent_data) + 1)))

      
    
    return fig


def ask_vector_query(query, top_results, ticker, date, pinecone_index, num_days_back, df_chunk, up_down):
    """
    Queries a vector database for relevant financial news articles based on a given query and filters.

    Args:
        query (str): The query string to embed and search against.
        top_results (int): The number of top results to return from the vector database.
        ticker (str): The stock ticker symbol to filter articles by.
        date (str or None): The specific date to filter articles by in 'YYYY-MM-DD' format. If None, no date filter is applied.
        pinecone_index (str): The name of the Pinecone index to query.
        num_days_back (int): The number of days back from the given date to include in the date filter.
        df_chunk (pd.DataFrame): DataFrame containing article chunks with 'Chunk_ID' and 'Text Chunks' columns.
        up_down (str or None): Indicator for filtering articles based on predicted stock movement ('up' or 'down'). If None, this filter is ignored.

    Returns:
        tuple: A tuple containing the generated response text and a list of article IDs from which text chunks were retrieved.
    """
    try:
        # Embed query
        query_embeddings = embedding_model.encode(query)
        print(f"Query Embeddings: {query_embeddings}")

        # Date filter for vector database
        if date is None:
            filter = {"ticker": {"$eq": ticker}}
        elif date and up_down:
            up_down_indicator = 1 if up_down == 'increase' else 0
            date_obj = date
            previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, num_days_back + 1)]
            filter = {
                "ticker": {"$eq": ticker},
                "date": {"$in": previous_week_dates},
                "up_down_prediction": {"$eq": str(up_down_indicator)}
            }
        else:
            date_obj = date
            previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, num_days_back + 1)]
            filter = {
                "ticker": {"$eq": ticker},
                "date": {"$in": previous_week_dates}
            }
        print(f"Filter: {filter}")

        # Run vector database query
        index = pc.Index(pinecone_index)
        output = index.query(
            namespace="ns1",
            vector=[float(i) for i in list(query_embeddings)],
            filter=filter,
            top_k=top_results,
            include_values=False,
            include_metadata=True
        )
        print(f"Query Output: {output['matches']}")

        # Process results
        article_ids = []
        retrieved_text = ' '
        for match in output['matches']:
            Chunk_UID = match['id']
            chunk_text = df_chunk[df_chunk['Chunk_ID'] == Chunk_UID]['Text Chunks'].values[0]
            print(f"Chunk Text: {chunk_text}")
            article_ids.append('-'.join(Chunk_UID.split('-')[:2]))
            retrieved_text += ' ' + chunk_text
        print(f"Article IDs: {article_ids}")
        print(f"Retrieved Text: {retrieved_text}")

        # Generate response
        query_context = f"""
            Given the text from a financial news article, analyze the content and produce a bulleted response to the provided query:
            **Constraints:** ONLY RESPOND USING THE PROVIDED Context
            The context from the financial news article excerpts below:
            {retrieved_text}
            Query:
            {query}
            Example Response:
            - **subject**: explanation
            - **subject**: explanation
            - **subject**: explanation
        """
        print(f"Query Context: {query_context}")

        response = model.generate_content(query_context)
        print(f"Generated Response: {response.text}")
        return response.text, article_ids

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, []



# Streamlit app layout
st.markdown("<h1 style='text-align: center;'>FAST OG</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Financial Analysis Sentiment Tool for Oil & Gas</h3>", unsafe_allow_html=True)
# Load CSV into a DataFrame
#csv_path = os.path.join(os.pardir, '02_Cleaned_Data', 'SEC_Filings.csv')
#sec_df = pd.read_csv(csv_path)

# Define the base directory relative to the current script location
# Define the base directory relative to the current script location
base_dir = os.path.abspath(os.path.dirname(__file__))
print('checking directory', base_dir)
# Helper function to load CSV files
def load_csv(file_path):
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        print(f"File does not exist: {full_path}")
        return pd.DataFrame()

# Load sentiment results
df_sentiment = load_csv('Prompt2_Sentiment_Analysis_Results.csv')




# Load Vector Full Article References
full_files = [
    'Article_Full_References_pt1.csv',
    'Article_Full_References_pt2.csv',
    'Article_Full_References_pt3.csv'
]
df_full = pd.concat([load_csv(file) for file in full_files], ignore_index=True)
df_full['Unique_ID'] = df_full['Chunk_ID'].apply(lambda x: '-'.join(x.split('-')[:2]))

# Load Article Headline and URL References
columns_to_load = ['Source', 'Unique_ID', 'Date', 'Article Headline', 'URL']
article_files = [
    'Investment_Research_Part1.csv',
    'Investment_Research_Part2.csv',
    'ProQuest_Articles.csv',
    'Earnings_Presentations.csv',
    'Earnings_QA.csv',
    'SEC_Filings.csv'
]
articles_dfs = [load_csv(file)[columns_to_load] for file in article_files]
articles_df = pd.concat(articles_dfs, ignore_index=True)


# Default tickers
default_tickers = ['BP', 'COP', 'CVX', 'CXO', 'DVN', 'EOG', 'EQNR', 'HES', 'MPC', 'MRO', 'OXY', 'PDCE', 'PSX', 'PXD', 'SHEL', 'TTE', 'VLO', 'XOM']

# Dropdown for tickers with blank default and single selection
ticker = st.selectbox('Select a stock ticker', options=[''] + default_tickers)

if ticker != '':
    st.session_state.selected_ticker = ticker
    st.session_state.plot_shown = True   

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
start_date = pd.to_datetime(articles_df['Date']).min()#datetime(2019, 1, 1)
end_date = pd.to_datetime(articles_df['Date']).max() - pd.Timedelta(days=5)#datetime.today()

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
            st.write(f'Daily stock price change on {selected_date}: **{abs(st.session_state.stock_change):.2%} {up_down}**')
        except:
            pass
        
        st.divider()
        
        try:
            st.markdown("<h2 style='text-align: center;'>Sentiment Counts for Last Week of Articles</h2>", unsafe_allow_html=True)

            num_days_back = 7
            # Slider for select top number of vector results, used for sensitivit testing
            #num_days_back = st.select_slider("Select Number of Days Back to Use", 
            #                                        options = list(range(0,15)),
            #                                        value = 7,
            #                                        key='num_days')
        
            # Draw Sentiment Plot
            fig_sent = plot_sentiment(df_sentiment, st.session_state.selected_ticker, selected_date, num_days_back)   
            st.plotly_chart(fig_sent)
        except:
            pass



# LLM Model setup.  (API Key needs to be in your environment variables)
key = 'AIzaSyC_hI1l9OTJhYoFw3UC-5LAfJXfENX9COs'
GOOGLE_API_KEY = os.getenv(key)
print(f"Google API Key: {GOOGLE_API_KEY}")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')
print("Successfully connected to Google Generative AI with key.")


       
# Query input and response
try:
    if selected_date != None and st.session_state.stock_change != None:
        #st.header('What potential factors caused ' + ticker + ' stock price to ' + up_down + ' on ' + str(selected_date) + '?')
        st.markdown("<h2 style='text-align: center;'>" + \
                    'What potential factors caused ' + ticker + ' stock price to ' + up_down + ' on ' + str(selected_date) + '?' + \
                    "</h2>", unsafe_allow_html=True)

        selected_chunk_count = 5
        # Slider for select top number of vector results, used for sensitivity testing
        #selected_chunk_count = st.select_slider("Select Number of References to Answer From", 
        #                                            options = list(range(3,11)), 
        #                                            key='select_chunk')
        
        #st.write('What potential factors caused ' + ticker + ' stock price to ' + up_down + ' on ' + str(selected_date) + '?')
        query = 'What potential factors caused ' + ticker + ' stock price to ' + up_down +'?'
        if st.button('Generate Response from Google Gemini') and query:
            response_vector, article_ids = ask_vector_query(query, selected_chunk_count, ticker, selected_date, "fastfullvectors", num_days_back, df_full, None)
            st.session_state.response_vector = response_vector.replace('$','\$')
            print(st.session_state.response_vector)
        try:
            st.write('Response:')
            st.write(st.session_state.response_vector)
            try:
                st.session_state.articles_df_filtered = articles_df[articles_df['Unique_ID'].isin(article_ids)]
            except:
                pass
            st.write('References:')
            for index, row in st.session_state.articles_df_filtered.iterrows():
                #st.write(f"Source: {row['Source']}, Article Headline: {row['Article Headline']}, Date: {row['Date']}, Article ID: {row['Unique_ID']}")
                if st.checkbox(f"Show text for: Source: {row['Source']}, Article Headline: {row['Article Headline']}, Date: {row['Date']}, Article ID: {row['Unique_ID']}", key=row['Unique_ID']):
                    with st.container(height=300):
                        st.markdown(df_full[df_full['Unique_ID'] == row['Unique_ID']]['Text Chunks'].iloc[0].replace('$','\$'))
        except:
            pass

except:
    pass

st.divider()

# Query input and response, comment out for demo
if st.session_state.response_vector:
    st.markdown("<h2 style='text-align: center;'>" + 'Custom Generative AI Query' + "</h2>", unsafe_allow_html=True)
    
    ask_query = st.text_input('Ask your own query:')

    st.write(f'Toggle on to use full database, toggle off to use articles from {selected_date-timedelta(days=7)} to {selected_date}')
    on = st.toggle("Use All Articles", value=True)
    if on:
        ask_selected_date = None
    else:
        ask_selected_date = selected_date
    
    selected_ask_chunk_count = 5
    # Slider for select top number of vector results, used for sensitivity testing
    #selected_ask_chunk_count = st.select_slider("Select Number of References to Answer From", 
    #                                            options = list(range(3,11)), 
    #                                            key='ask_chunk')
    if st.button('Ask Question') and ask_query:
        ask_response, ask_article_ids = ask_vector_query(ask_query, selected_ask_chunk_count, ticker, ask_selected_date, "fastfullvectors", num_days_back, df_full, None)
        st.session_state.ask_response = ask_response
    try:
        st.write('Response:')
        st.write(st.session_state.ask_response)
        st.write(articles_df[articles_df['Unique_ID'].isin(ask_article_ids)])
    except:
        pass
