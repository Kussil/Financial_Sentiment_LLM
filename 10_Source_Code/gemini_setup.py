# gemini_setup.py

import os
import pandas as pd
import re
import google.generativeai as genai

def configure_gemini():
    """
    Configure the Gemini model using the API key from the environment variables and set safety settings.

    Returns:
        model: Configured Gemini model.
    """
    key = 'GOOGLE_API_KEY'
    GOOGLE_API_KEY = os.getenv(key)
    genai.configure(api_key=GOOGLE_API_KEY)

    # Safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }
    ]

    model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)
    return model

def find_first_unique_id_with_empty_values(file_path, categories):
    """
    Finds the first unique ID where any of the specified columns have empty values in a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        categories (list of str): List of column names to check for empty values.

    Returns:
        str: The first Unique_ID where any of the specified columns have empty values.
        None: If no such row is found.
    """
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        if row[categories].isnull().any() or (row[categories] == '').any():
            return row['Unique_ID']
    return None

def get_gemini_inputs(text_df, unique_id):
    """
    Retrieves information from the DataFrame based on the unique ID and outputs company, source, headline, and text.

    Args:
        text_df (pd.DataFrame): The DataFrame containing the text data.
        unique_id (str): The unique ID to search for.

    Returns:
        tuple: A tuple containing company, source, headline, and text.
    """
    row = text_df[text_df['Unique_ID'] == unique_id]
    company = row['Ticker'].values[0]
    source = row['Source'].values[0]
    headline = row['Article Headline'].values[0]
    text = row['Article Text'].values[0]
    return company, source, headline, text

def query_gemini(prompt, model):
    """
    Query Gemini to perform sentiment analysis on text from various sources about a company.

    Args:
        prompt (str): The prompt to be used for querying the model.
        model: The model object used to generate content and analyze the text.

    Returns:
        str: The sentiment analysis results for predefined categories in the specified format.
    """
    response = model.generate_content(prompt)
    return response.text

def parse_sentiment(text, categories):
    """
    Parses a given text for specified categories and their sentiments.

    Args:
        text (str): The input text containing categories and their sentiments.
        categories (list of str): List of category names to look for in the text.

    Returns:
        dict or str: A dictionary with categories as keys and their corresponding sentiments as values,
                     or "Did not find all categories" if any sentiment is not Positive, Neutral, or Negative.
    """
    results = {}
    valid_sentiments = {"Positive", "Neutral", "Negative"}

    for category in categories:
        pattern = rf"- {re.escape(category)} - (\w+)"
        match = re.search(pattern, text)
        if match:
            sentiment = match.group(1)
            if sentiment not in valid_sentiments:
                return "Did not find all categories"
            results[category] = sentiment
        else:
            return "Did not find all categories"

    return results

def update_csv(file_path, unique_id, sentiment_dict):
    """
    Updates the columns of a CSV file based on the unique ID and sentiment dictionary.

    Args:
        file_path (str): The path to the CSV file.
        unique_id (str): The unique ID of the row to be updated.
        sentiment_dict (dict): A dictionary with categories as keys and their corresponding sentiments as values.

    Returns:
        None
    """
    df = pd.read_csv(file_path)
    row_index = df[df['Unique_ID'] == unique_id].index
    for category, sentiment in sentiment_dict.items():
        df.loc[row_index, category] = sentiment
    df.to_csv(file_path, index=False)
    print(f"Row with Unique_ID '{unique_id}' has been updated.")
