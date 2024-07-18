# 07_User_Interface Folder

This folder contains application user interface that utilized the article text, sentiment analysis, stock modeling, and vector database to implement a Retreival Augmented Generation (RAG) application to derive insights about stock price movement from relevant articles.

## Directory Contents
- **app_demo.py**

## Running the Application
To run the Streamlit application, execute the following command in your terminal:

```sh
streamlit run app_demo.py
```
This command will start a local web server, and you can view the application by navigating to http://localhost:8501 in your web browser.

## How to Use
1. Ensure that you have GOOGLE_API_KEY for Gemini set as an environment variable. To get an API key you can go to: <https://ai.google.dev/gemini-api/docs/api-key>.
2. Select a date from the graph by clicking or using the dropdown select box. This will display the stock price change for that day.
3. The sentiments for the last 7 days will be summarized in a chart showing the sentiments for each article for each sentiment category.
4. Click the Generate Response button to get a list of insights to the stock price change based on articles from the last 7 days.
