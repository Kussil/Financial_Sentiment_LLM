# 02_Cleaned_Data Folder

This folder contains the cleaned datasets and preprocessing script for the Financial Sentiment LLM project. These cleaned data files will be used by the notebooks in the 03_Sentiment_Analysis folder. Below is a brief description of each file and its purpose.

## Directory Contents

### CSV Files
- **Earnings_Presentations.csv**
- **Earnings_QA.csv**
- **Investment_Research_Part1.csv**
- **Investment_Research_Part2.csv**
- **ProQuest_Articles.csv**
- **SEC_Filings.csv**

All CSV files have been cleaned and organized to follow a uniform tabular structure. The Investment Research files are split due to GitHub file size constraints. Notebooks in the 03_Sentiment_Analysis folder will concatenate ALL of these files into a single dataframe.

### Jupyter Notebook
- **Text_Preprocessing.ipynb**

This Jupyter Notebook handles the text preprocessing of the datasets. It includes various steps for cleaning and standardizing the data from different sources. Each data source is processed individually according to its requirements, but all are cleaned to a consistent standard.