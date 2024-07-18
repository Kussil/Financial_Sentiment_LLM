
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
    fig = px.line(stock_data, 
                  x=stock_data.index, 
                  y='Close', 
                  title=f'Stock Prices for {ticker}')
    fig.update_layout(title=f'Stock Prices for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Stock Price (USD)')
    fig.update_traces(hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y}')
    try:
        selected_points = plotly_events(fig)
        graph_date = selected_points[0]['x']
        date_object = datetime.strptime(graph_date, "%Y-%m-%d")
        st.session_state.stock_change = stock_data['Change'].loc[graph_date]
        
        
    except:
        date_object = end_date
    
    # set date_object to trigger query
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
    previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%#m/%#d/%Y') for i in range(0, num_days_back + 1)]
    
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
    summary_melted = pd.melt(summary, id_vars=['index'], var_name='Sentiment', value_name='Count')

    # Plotting with Plotly Express
    fig = px.bar(summary_melted, x='Count', y='index', color='Sentiment', orientation='h',
                 labels={'index': 'Category', 'Count': 'Count', 'Sentiment': 'Sentiment'},
                 title='Sentiment Counts',
                 barmode='stack',
                 color_discrete_map = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'})
    
    fig.update_layout(title={'text': 'Sentiment Counts',
                         'x': 0.5,  # Center align title horizontally
                         'y': 0.95,  # Position title slightly above the plot
                         'xanchor': 'center',  # Center align title horizontally
                         'yanchor': 'top'})  # Position title slightly above the plot

      
    
    return fig


def ask_vector_query(query, top_results, ticker, date, pinecone_index, num_days_back):
    """
    Queries a vector database with an embedded query and retrieves relevant text chunks based on filters.
    
    Parameters:
    query (str): The query string to be embedded and searched in the vector database.
    top_results (int): The number of top results to retrieve from the vector database.
    ticker (str): The stock ticker to filter the search results.
    date (str or None): The date to filter the search results. If None, no date filter is applied.
    pinecone_index (str, optional): The name of the Pinecone index to use. Default is "fastvectors".
    
    Returns:
    str: A generated response text based on the context retrieved from the vector database.
    
    Notes:
    - The function first embeds the query using the `embedding_model`.
    - If a date is provided, it generates a list of dates for the previous week to filter the search results.
    - It queries the Pinecone vector database with the embedded query and the filters.
    - The function retrieves and concatenates the relevant text chunks from the database.
    - It then generates a response based on the retrieved context and the original query.
    """
    # embed query
    query_embeddings = embedding_model.encode(query)

    # date filter for vector database
    if date == None:
        # If no date passed exclude date filter
        filter={"ticker": {"$eq": ticker}}
        
    else:
        # Convert the string to a datetime object
        date_obj = date#datetime.strptime(date, '%Y-%m-%d')

        # Generate the list of dates for the previous week
        previous_week_dates = [(date_obj - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, num_days_back + 1)]
        print(previous_week_dates)
        
        # Create Filter
        filter={
            "ticker": {"$eq": ticker},
            "date": {"$in": previous_week_dates}
        }

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
    print(output['matches'])
    
    article_ids = []
    retrieved_text = ' '
    for i in range(len(output['matches'])):
        Chunk_UID = output['matches'][i]['id']
        chunk_text = df_chunk[df_chunk['Chunk_ID']==Chunk_UID]['Text Chunks'].values[0]
        print('#########################################')
        print(chunk_text)
        article_ids.append('-'.join(Chunk_UID.split('-')[:2]))
        retrieved_text += ' ' + chunk_text
    
    print(article_ids)
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
    return response.text, article_ids


# Streamlit app layout
st.title('FAST OG: Stock Price Analyzer')

# Load CSV into a DataFrame
#csv_path = os.path.join(os.pardir, '02_Cleaned_Data', 'SEC_Filings.csv')
#sec_df = pd.read_csv(csv_path)

# Load sentiment results
df_sentiment = pd.read_csv(os.path.join(os.pardir,'03_Sentiment_Analysis', 'Gemini', 'Prompt2', 'Prompt2_Sentiment_Analysis_Results.csv'))

# Load Vector Chunk References
df1_chunk = pd.read_csv(os.path.join(os.pardir, '05_Create_Vector_DB', 'Gemini', 'Article_Chunk_References_pt1.csv'))
df2_chunk = pd.read_csv(os.path.join(os.pardir, '05_Create_Vector_DB', 'Gemini', 'Article_Chunk_References_pt2.csv'))
df3_chunk = pd.read_csv(os.path.join(os.pardir, '05_Create_Vector_DB', 'Gemini', 'Article_Chunk_References_pt3.csv'))
df_chunk = pd.concat([df1_chunk, df2_chunk, df3_chunk], ignore_index=True)

# Load Article Headline and URL References
columns_to_load = ['Source', 'Unique_ID', 'Date', 'Article Headline', 'URL']
invest_df1 = pd.read_csv(os.path.join(os.pardir, '02_Cleaned_Data', 'Investment_Research_Part1.csv'), usecols=columns_to_load)
invest_df2 = pd.read_csv(os.path.join(os.pardir, '02_Cleaned_Data', 'Investment_Research_Part2.csv'), usecols=columns_to_load)
proquest_df = pd.read_csv(os.path.join(os.pardir, '02_Cleaned_Data', 'ProQuest_Articles.csv'), usecols=columns_to_load)
earnings_presentations = pd.read_csv(os.path.join(os.pardir, '02_Cleaned_Data', 'Earnings_Presentations.csv'), usecols=columns_to_load)
earnings_qa = pd.read_csv(os.path.join(os.pardir, '02_Cleaned_Data', 'Earnings_QA.csv'), usecols=columns_to_load)
sec_df = pd.read_csv(os.path.join(os.pardir, '02_Cleaned_Data', 'SEC_Filings.csv'), usecols=columns_to_load)
articles_df = pd.concat([invest_df1, invest_df2, proquest_df, sec_df, earnings_presentations, earnings_qa], ignore_index=True)

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
            st.write(f'Stock change for {selected_date} was {st.session_state.stock_change:.0%}')
        except:
            pass
        
        st.divider()
        
        try:
            st.header('Sentiment Summary')

            num_days_back = 7
            # Slider for select top number of vector results, used for sensitivit testing
            #num_days_back = st.select_slider("Select Number of Days Back to Use", 
            #                                        options = list(range(0,15)),
            #                                        value = 7,
            #                                        key='num_days')
        
            # Draw Sentiment Plot
            fig_sent = plot_sentiment(df_sentiment, st.session_state.selected_ticker, date_object, num_days_back)   
            st.plotly_chart(fig_sent)
        except:
            pass



# LLM Model setup.  (API Key needs to be in your environment variables)
key = 'GOOGLE_API_KEY'
GOOGLE_API_KEY = os.getenv(key)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

       
# Query input and response
try:
    if st.session_state.date_object and st.session_state.stock_change != None:
        st.header('Stock Query')
        
        selected_chunk_count = 5
        # Slider for select top number of vector results, used for sensitivity testing
        #selected_chunk_count = st.select_slider("Select Number of References to Answer From", 
        #                                            options = list(range(3,11)), 
        #                                            key='select_chunk')
        
        # Create up or down indictor of stock price change for query
        if st.session_state.stock_change > 0:
            up_down = 'up'
        else:
            up_down = 'down'
        
        st.write('What is impacting ' + ticker + ' stock price to go ' + up_down +'?')
        query = 'What is impacting ' + ticker + ' stock price' + up_down +'?'
        if st.button('Generate Response') and query:
            response_vector, article_ids = ask_vector_query(query, selected_chunk_count, ticker, selected_date, "fastvectors", num_days_back)
            st.session_state.response_vector = response_vector
            print(st.session_state.response_vector)
        try:
            st.write('Response:')
            st.write(st.session_state.response_vector)
            st.write(articles_df[articles_df['Unique_ID'].isin(article_ids)])
        except:
            pass
except:
    pass

#st.divider()

# Query input and response, comment out for demo
#if st.session_state.response_vector:
#    st.header('Custom Query')
#    
#    st.write('Ask your own question about articles:')
#    ask_query = st.text_input('Enter your query:')
#    
#    on = st.toggle("Used Selected Date", value=True)
#    if on:
#        ask_selected_date = selected_date
#    else:
#        ask_selected_date = None
#    
#    selected_ask_chunk_count = 5
#    # Slider for select top number of vector results, used for sensitivity testing
#    #selected_ask_chunk_count = st.select_slider("Select Number of References to Answer From", 
#    #                                            options = list(range(3,11)), 
#    #                                            key='ask_chunk')
#    if st.button('Ask Question') and ask_query:
#        ask_response, ask_article_ids = ask_vector_query(ask_query, selected_ask_chunk_count, ticker, ask_selected_date, "fastvectors", num_days_back)
#        st.session_state.ask_response = ask_response
#    try:
#        st.write('Ask Response:')
#        st.write(st.session_state.ask_response)
#        st.write(articles_df[articles_df['Unique_ID'].isin(ask_article_ids)].style.hide_index())
#    except:
#        pass