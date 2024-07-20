"""
This module provides functions to load, clean, and process data for sentiment analysis.

Functions:
- load_cleaned_data: Load and merge cleaned data files into a single DataFrame.
- drop_unprocessable_rows: Drop rows from the DataFrame that cannot be processed.
- check_file_exists: Check if a file exists at the specified file path.
- create_empty_sentiment_df: Create an empty DataFrame for sentiment analysis results.
- save_dataframe_to_csv: Save a DataFrame to a CSV file.
- main: Main function to load data, process it, and save the results.

Usage:
Import this module and call the functions as needed, or run the script directly.
"""

import os
import pandas as pd
import re
import json

def load_cleaned_data() -> pd.DataFrame:
    """
    Load and merge cleaned data files into a single DataFrame.

    Returns:
        pd.DataFrame: The merged DataFrame containing all the cleaned data.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths for the CSV files
    invest_df1_path = os.path.join(script_dir, '../02_Cleaned_Data/Investment_Research_Part1.csv')
    invest_df2_path = os.path.join(script_dir, '../02_Cleaned_Data/Investment_Research_Part2.csv')
    proquest_df_path = os.path.join(script_dir, '../02_Cleaned_Data/ProQuest_Articles.csv')
    earnings_presentations_path = os.path.join(script_dir, '../02_Cleaned_Data/Earnings_Presentations.csv')
    earnings_qa_path = os.path.join(script_dir, '../02_Cleaned_Data/Earnings_QA.csv')
    sec_df_path = os.path.join(script_dir, '../02_Cleaned_Data/SEC_Filings.csv')
    
    # Load the CSV files
    invest_df1 = pd.read_csv(invest_df1_path)
    invest_df2 = pd.read_csv(invest_df2_path)
    proquest_df = pd.read_csv(proquest_df_path)
    earnings_presentations = pd.read_csv(earnings_presentations_path)
    earnings_qa = pd.read_csv(earnings_qa_path)
    sec_df = pd.read_csv(sec_df_path)

    # Concatenate the DataFrames
    text_df = pd.concat([
        invest_df1, invest_df2, proquest_df, sec_df,
        earnings_presentations, earnings_qa
    ], ignore_index=True)
    
    return text_df

def drop_unprocessable_rows(df: pd.DataFrame, rows_to_drop: list) -> pd.DataFrame:
    """
    Drop rows from the DataFrame that cannot be processed.

    Args:
        df (pd.DataFrame): The input DataFrame.
        rows_to_drop (list): List of Unique_IDs to be dropped.

    Returns:
        pd.DataFrame: DataFrame with specified rows dropped.
    """
    index_to_drops = df[df['Unique_ID'].isin(rows_to_drop)].index
    df.drop(index_to_drops, inplace=True)
    return df

def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists at the specified file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def create_empty_sentiment_df(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    Create an empty DataFrame for sentiment analysis results with specified categories.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categories (list): List of sentiment categories.

    Returns:
        pd.DataFrame: DataFrame prepared for sentiment analysis results.
    """
    sentiment_df = df.copy()
    sentiment_df.drop(['Article Text', 'Article Headline'], axis=1, inplace=True)

    for category in categories:
        sentiment_df[category] = ""
        sentiment_df[category] = sentiment_df[category].astype('object')

    return sentiment_df

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output CSV file.
    """
    df.to_csv(file_path, index=False)

def main(sentiment_results_file_path: str, rows_to_drop: list):
    """
    Main function to load data, process it, and save the results.

    Args:
        sentiment_results_file_path (str): The file path to save the sentiment analysis results.
        rows_to_drop (list): List of Unique_IDs to be dropped.
    """
    # Load and merge data
    text_df = load_cleaned_data()
    
    # Drop unprocessable rows
    text_df = drop_unprocessable_rows(text_df, rows_to_drop)
    
    # Check if sentiment analysis results file exists
    file_exists = check_file_exists(sentiment_results_file_path)

    if file_exists:
        print(f"The file exists in the current directory.")
    else:
        print(f"The file does not exist in the current directory.")
        empty_sentiment_df = create_empty_sentiment_df(text_df, CATEGORIES)
        save_dataframe_to_csv(empty_sentiment_df, sentiment_results_file_path)

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

def get_model_inputs(text_df, unique_id):
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
    valid_sentiments = {"Positive", "Neutral", "Negative", 'Not mentioned'}

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


def extract_and_convert_to_dict(response):
    # Regular expression to find the JSON object within curly braces
    match = re.search(r'\{[^}]*\}', response, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        return "No JSON object found in the response."
    
    # Replace single quotes with double quotes
    json_str = re.sub(r"(?<!\\)'", '"', json_str)

    # Handle escaped single quotes within the string
    json_str = json_str.replace("\\'", "'")

    # Parse the JSON string into a dictionary
    try:
        parsed_response = json.loads(json_str)
        return parsed_response
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        return None

def update_csv(file_path, unique_id, sentiment_dict, categories):
    """
    Updates the columns of a CSV file based on the unique ID and sentiment dictionary.

    Args:
        file_path (str): The path to the CSV file.
        unique_id (str): The unique ID of the row to be updated.
        sentiment_dict (dict): A dictionary with categories as keys and their corresponding sentiments as values.
        categories (list): List of all possible categories.

    Returns:
        None
    """
    df = pd.read_csv(file_path)
    row_index = df[df['Unique_ID'] == unique_id].index
    for category in categories:
        sentiment = sentiment_dict.get(category, "No JSON found")
        if sentiment == "N/A":
            sentiment = "Neutral"
        df.loc[row_index, category] = sentiment
    df.to_csv(file_path, index=False)
    print(f"Row with Unique_ID '{unique_id}' has been updated.")




if __name__ == "__main__":
    default_sentiment_results_file_path = 'Prompt2_Consistency_Check_Sentiment_Analysis_Results.csv'
    default_rows_to_drop = ['PQ-2840736837']
    main(default_sentiment_results_file_path, default_rows_to_drop)