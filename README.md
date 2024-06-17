# Financial Analysis Sentiment Tool
![Logo](09_Miscellaneous/Project_logo_ReadMe.png)


## Table of Contents
1. [Description](#description)
2. [Demo](#demo)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Installation Guidance](#installation)
6. [How to work using Google Colab](#using-github-repositories-in-google-colab)

## Description
This project aims to analyze the dynamics of stock prices for around 20 industry-related competitors, such as Chevron, Exxon, and Shell. The analysis is based on various factors, including news sentiment, financial reports, and other documents. The project combines fundamental and technical analysis methodologies to predict stock price movements and understand market behavior.

The specific goals include:
- Developing a News Sentiment Analysis Tool: Analyzing sentiment from news articles, financial reports, and extensive documents.
- Determining Feature Importance to Predict Stock Price Movement: Identifying key factors that significantly influence stock price movements.
- Explaining Stock Price Movements Using News Articles: Finding relevant documents and references explaining stock price movements.
- Delivering a Comprehensive Analysis Report: Producing detailed reports summarizing the findings.

## Demo
Here’s a quick demo of the Financial Analysis Sentiment Tool in action:

![Demo](09_Miscellaneous/Demo.gif)

To run the demo using Streamlit, follow these steps:

1. Make sure you have all dependencies installed.
2. In your terminal, navigate to the project directory.
3. Run the following command:
    ```bash
    streamlit run app_with_click.py
    ```
4. A new tab will open in your default web browser showing the demo interface.

## Features
- **Sentiment Analysis**: Analyzes the sentiment of financial news articles and reports, categorizing them as positive, negative, or neutral.
- **Feature Importance Analysis**: Identifies the key factors (features) that significantly influence stock price movements.
- **Stock Price Movement Summaries**: Generates summaries based on news and financial reports, explaining stock price movements and providing relevant references.
- **Google Colab Integration**: Allows easy access and collaboration through Google Colab, enabling users to run notebooks and access files directly from GitHub.


## Project Structure
The project is organized into the following folders:

- `00_Temp`: Temporary files and data.
- `01_Raw_Data`: Raw data collected for analysis.
- `02_Cleaned_Data`: Cleaned and processed data.
- `03_Sentiment_Analysis`: Scripts and notebooks for sentiment analysis.
- `04_Stock_Modeling`: Stock modeling and prediction scripts.
- `05_Create_Vector_DB`: Scripts to create vector databases.
- `06_Query_Vector_DB`: Querying the vector databases.
- `07_User_Interface`: User interface components. 
- `08_Presentations`: Presentations and related materials.
- `09_Miscellaneous`: Miscellaneous files, including the project logo, references etc.

## Installation
To install and set up the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file and add necessary environment variables.
    
    ```text
    API_KEY=your_api_key
    OTHER_VARIABLE=value
    ```

## Using GitHub Repositories in Google Colab

This guide explains how to link and use GitHub repositories in Google Colab, enabling you to run notebooks and access files directly from GitHub.

### Step 1: Open Google Colab

Access Google Colab by visiting [Google Colab](https://colab.research.google.com/) and start a new notebook.

### Step 2: Linking GitHub Repository

To use notebooks from GitHub:
- In your Colab notebook, navigate to `File` > `Open notebook`.
- Switch to the "GitHub" tab in the dialog that appears.
- Paste the URL of the GitHub repository or enter the username to browse repositories.
- Choose the desired notebook file (`.ipynb`) from the repository.

### Step 3: Accessing Data from the Repository
After cloning, access files using:
import pandas as pd
data = pd.read_csv('/content/repository/datafile.csv')

### Step 4: Saving Your Notebook
To save changes back to GitHub:
- Choose File > Save a copy in GitHub.
- Authorize Colab if prompted.
- Select the target repository and branch.
- Enter a commit message and click "OK".
