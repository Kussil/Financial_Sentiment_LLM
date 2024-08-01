# Financial Analysis Sentiment Tool

![Logo](09_Miscellaneous/logo_3.png)

## Table of Contents


- [Project Description](#project-description)
- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setting Up the Project](#setting-up-the-project)
   - [Create a Virtual Environment](#1-create-a-virtual-environment)
   - [Activate the Virtual Environment](#2-activate-the-virtual-environment)
   - [Install Dependencies](#3-install-dependencies)
   - [Deactivate the Virtual Environment](#4-deactivate-the-virtual-environment)
   - [Obtain API Keys](#5-obtain-api-keys)
- [Project Report](#project-report)
- [Authors and Contact Information](#authors-and-contact-information)



## Project Description

This project aims to develop a News Sentiment Analysis Tool for analyzing sentiment from news articles, financial reports, and extensive documents related to the selected oil companies. Another key objective is to determine feature importance to predict stock price movement by identifying key factors that significantly influence stock price movements in the oil industry. Additionally, the project seeks to explain stock price movements using news articles by finding relevant documents and references that elucidate observed stock price fluctuations.


## Demo
Here’s a quick demo of the Financial Analysis Sentiment Tool in action:


![Demo](09_Miscellaneous/Demo.gif)

To run the demo using Streamlit, follow these steps:

1. Make sure you have all dependencies installed.
2. In your terminal, navigate to the project directory.
3. Run the following command:
    ```bash
    streamlit run app_demo.py
    ```
4. A new tab will open in your default web browser showing the demo interface.

Or simply follow the link: https://rice-fast-og.purplestone-988dcb3b.eastus.azurecontainerapps.io
## Features
- **Sentiment Analysis**: Analyzes the sentiment of financial news articles and reports, categorizing them as positive, negative, or neutral.
- **Feature Importance Analysis**: Identifies the key factors (features) that significantly influence stock price movements.
- **Stock Price Movement Summaries**: Generates summaries based on news and financial reports, explaining stock price movements and providing relevant references.
- **Google Colab Integration**: Allows easy access and collaboration through Google Colab, enabling users to run notebooks and access files directly from GitHub.
- **Web App**: Access the Financial Analysis Sentiment Tool through an interactive web application for sentiment analysis and stock price movement summaries.

## Project Structure
The project is organized into the following folders:

- `00_Temp`: Temporary working files not critical to the main repository.
- `01_Raw_Data`: Data scraping notebooks and resulting raw data files.
- `02_Cleaned_Data`: Cleaning notebook and cleaned and processed data.
- `03_Sentiment_Analysis`: Files and notebooks for sentiment analysis.
- `04_Stock_Modeling`: Stock modeling notebooks.
- `05_Create_Vector_DB`: Notebooks to create vector our database.
- `06_Query_Vector_DB`: Notebook to test querying the vector database.
- `07_User_Interface`: Script to run our streamlit app. 
- `08_Presentations`: Presentations and related materials.
- `09_Miscellaneous`: Miscellaneous files, including the project logo, references etc.
- `10_Source_Code`: Files with helper functions

## Setting Up the Project

### 1. Create a Virtual Environment

<table>
  <tr>
    <th>Platform</th>
    <th>Command</th>
  </tr>
  <tr>
    <td>Linux/macOS</td>
    <td><code>bash setup_env.sh</code></td>
  </tr>
  <tr>
    <td>Windows</td>
    <td><code>setup_env.bat</code></td>
  </tr>
</table>

### 2. Activate the Virtual Environment

<table>
  <tr>
    <th>Platform</th>
    <th>Command</th>
  </tr>
  <tr>
    <td>Linux/macOS</td>
    <td><code>source env/bin/activate</code></td>
  </tr>
  <tr>
    <td>Windows</td>
    <td><code>.\env\Scripts\activate</code></td>
  </tr>
</table>

### 3. Install Dependencies

After installing the virtual environment, run the following command to install all required dependencies. This command is the same for both Windows and Linux/macOS platforms:

```bash
pip install -r requirements.txt
```

### 4. Deactivate the Virtual Environment

When you're done working, you can deactivate the virtual environment with:
```bash
deactivate
```

### 5. Obtain API Keys
All our notebooks are designed to run in Colab unless otherwise specified.  Notebooks in the 01_Raw_Data folder and the script in the 07_User_Interface folder are the primary exceptions.  While other folders may contain notebooks designed to run on a desktop, they will always have a Colab alternative.  The API keys below need to be input into the Colab secrets tab and also updated in the script in the 07_User_Interface folder.

1. **Gemini API key** - Follow the instructions [here](https://www.gemini.com/cryptopedia/api).
2. **Hugging Face API key** - Follow the instructions [here](https://huggingface.co/docs/api-inference/quicktour#getting-started).
3. **GitHub API key** - Follow the instructions [here](https://docs.github.com/en/rest/overview/other-authentication-methods#personal-access-tokens).

### 6. Project Report

For a detailed understanding of the project, including methodologies, results, and analysis, you can refer to the project report:

[Financial LLM Capstone Report](08_Presentations/Financial_LLM_Capstone.pdf)


### 7. Authors


**Authors:**
- Ben Weideman
- Danny Boone
- John Lattal
- Ilyas Kussanov

For any questions, suggestions, or collaboration inquiries, feel free to contact us at:

- **Email:** bw58@rice.edu, db101@rice.edu, jl351@rice.edu, ik25@rice.edu,
