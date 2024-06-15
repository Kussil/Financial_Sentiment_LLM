# Financial Analysis Sentiment Tool
![Logo](09_Miscellaneous/Project_logo_ReadMe.png)
## Table of Contents
1. [Description](#description)
2. [Running](#using-github-repositories-in-google-colab)
## Description
This project aims to analyze the dynamics of stock prices for around 20 industry-related competitors, such as Chevron, Exxon, and Shell. The analysis is based on various factors, including news sentiment, financial reports, and other documents. The project combines fundamental and technical analysis methodologies to predict stock price movements and understand market behavior.

The specific goals include:
- Developing a News Sentiment Analysis Tool: Analyzing sentiment from news articles, financial reports, and extensive documents.
- Determining Feature Importance to Predict Stock Price Movement: Identifying key factors that significantly influence stock price movements.
- Explaining Stock Price Movements Using News Articles: Finding relevant documents and references explaining stock price movements.
- Delivering a Comprehensive Analysis Report: Producing detailed reports summarizing the findings.


## Using GitHub Repositories in Google Colab

This guide explains how to link and use GitHub repositories in Google Colab, enabling you to run notebooks and access files directly from GitHub.

## Step 1: Open Google Colab

Access Google Colab by visiting [Google Colab](https://colab.research.google.com/) and start a new notebook.

## Step 2: Linking GitHub Repository

To use notebooks from GitHub:
- In your Colab notebook, navigate to `File` > `Open notebook`.
- Switch to the "GitHub" tab in the dialog that appears.
- Paste the URL of the GitHub repository or enter the username to browse repositories.
- Choose the desired notebook file (`.ipynb`) from the repository.

## Step 3: Accessing Data from the Repository
After cloning, access files using:
import pandas as pd
data = pd.read_csv('/content/repository/datafile.csv')

## Step 4: Saving Your Notebook
To save changes back to GitHub:
- Choose File > Save a copy in GitHub.
- Authorize Colab if prompted.
- Select the target repository and branch.
- Enter a commit message and click "OK".
