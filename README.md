# Financial Analysis Sentiment Tool
![Logo](09_Miscellaneous/logo_3.png)

## Table of Contents
1. [Description](#description)
2. [Demo](#demo)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Installation Guidance](#setting-up-the-project)

## Description
This project aims to develop a News Sentiment Analysis Tool for analyzing sentiment from news articles, financial reports, and extensive documents related to the selected oil companies. Another key objective is to determine feature importance to predict stock price movement by identifying key factors that significantly influence stock price movements in the oil industry. Additionally, the project seeks to explain stock price movements using news articles by finding relevant documents and references that elucidate observed stock price fluctuations.

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
- `10_Source_Code`: Miscellaneous files, including the project logo, references etc.

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

1. **Gemini API key** - Obtain a key to access the Gemini API. Follow the instructions [here](https://www.gemini.com/cryptopedia/api).
2. **Hugging Face API key** - Obtain a key to access Hugging Face models. Follow the instructions [here](https://huggingface.co/docs/api-inference/quicktour#getting-started).
3. **GitHub API key** - Obtain a key to access GitHub API. Follow the instructions [here](https://docs.github.com/en/rest/overview/other-authentication-methods#personal-access-tokens).

# Table of Contents

1. [.vscode](#vscode)
   - [settings.json](#settingsjson)

2. [00_Temp](#00_temp)
   - [Other](#other)
     - [API Costs.xlsx](#api-costsxlsx)
   - [Test_Notebooks](#test_notebooks)
     - [Mistral7b_Test.ipynb](#mistral7b_testipynb)
     - [Llama3_Test_V1.ipynb](#llama3_test_v1ipynb)
     - [Vector Database_Test.ipynb](#vector-database_testipynb)
     - [InteractivePlot_Test.ipynb](#interactiveplot_testipynb)
     - [Test_Notebooks](#test_notebooks_inner)
       - [Mistral7b_Test.ipynb](#mistral7b_testipynb_inner)
       - [Llama3_Test_V1.ipynb](#llama3_test_v1ipynb_inner)
       - [Vector Database_Test.ipynb](#vector-database_testipynb_inner)
       - [InteractivePlot_Test.ipynb](#interactiveplot_testipynb_inner)
       - [Sentiment Analysis.ipynb](#sentiment-analysisipynb)
       - [Mistral7b_Test_V2.ipynb](#mistral7b_test_v2ipynb_inner)
     - [SEC_URL_TextExtraction_TEST.ipynb](#sec_url_textextraction_testipynb)
     - [Mistral7b_Test_V2.ipynb](#mistral7b_test_v2ipynb)
     - [Sentiment_Analysis_Llama3.ipynb](#sentiment_analysis_llama3ipynb)
   - [Sentiment_Framework](#sentiment_framework)
     - [Category and Sentiment Framework.xlsx](#category-and-sentiment-frameworkxlsx)
     - [Sentiment_Framework](#sentiment_framework_inner)
       - [Category and Sentiment Framework.xlsx](#category-and-sentiment-frameworkxlsx_inner)
       - [Sentiment Comparisons.xlsx](#sentiment-comparisonsxlsx)
       - [Category and Sentiment Prompts.docx](#category-and-sentiment-promptsdocx)
     - [Sentiment Comparisons.xlsx](#sentiment-comparisonsxlsx_outer)
     - [Category and Sentiment Prompts.docx](#category-and-sentiment-promptsdocx_outer)
   - [Prompt_Engineering](#prompt_engineering)
     - [config.py](#configpy)
     - [llm_chain.py](#llm_chainpy)
     - [model.py](#modelpy)
     - [README.md](#readmemd)
     - [templates.py](#templatespy)
     - [Testing_notebook.ipynb](#testing_notebookipynb)
     - [utils.py](#utilspy)
     - [main.py](#mainpy)
     - [huggingface_login.py](#huggingface_loginpy)

3. [03_Sentiment_Analysis](#03_sentiment_analysis)
   - [finbert_sentiment_chunkdata_pt3.csv](#finbert_sentiment_chunkdata_pt3csv)
   - [finbert_sentiment_chunkdata_pt2.csv](#finbert_sentiment_chunkdata_pt2csv)
   - [Prompt2_Consistency_Check_Sentiment_Analysis_Gemini_Desktop.ipynb](#prompt2_consistency_check_sentiment_analysis_gemini_desktopipynb)
   - [finbert_sentiment_chunkdata_pt1.csv](#finbert_sentiment_chunkdata_pt1csv)
   - [Prompt1_Bias_Check_Sentiment_Analysis_Results.csv](#prompt1_bias_check_sentiment_analysis_resultscsv)
   - [Prompt2_Sentiment_Analysis_Results.csv](#prompt2_sentiment_analysis_resultscsv)
   - [Prompt2_Sentiment_Analysis_Gemini_Desktop.ipynb](#prompt2_sentiment_analysis_gemini_desktopipynb)
   - [finbert_sentiment_data.csv](#finbert_sentiment_datacsv)
   - [Sentiment_QC](#sentiment_qc)
     - [Sentiment_QC_Gemini_Ben.ipynb](#sentiment_qc_gemini_benipynb)
     - [Sentiment_QC_Gemini_Danny.ipynb](#sentiment_qc_gemini_dannyipynb)
     - [Sentiment_QC_Gemini_Ilyas.ipynb](#sentiment_qc_gemini_ilyasipynb)
     - [Sentiment_QC_Gemini_John.ipynb](#sentiment_qc_gemini_johnipynb)
     - [Sentiment_QC_Gemini.ipynb](#sentiment_qc_geminiipynb)
   - [Prompt1_Sentiment_Analysis_Gemini_Desktop.ipynb](#prompt1_sentiment_analysis_gemini_desktopipynb)
   - [Prompt2_Consistency_Check_Visualization.ipynb](#prompt2_consistency_check_visualizationipynb)
   - [Revised_notebooks](#revised_notebooks)
     - [Prompt_local_Llama.ipynb](#prompt_local_llamaipynb)
     - [Rev_1.2_Prompt_Sentiment_Analysis_Gemini_Desktop.ipynb](#rev_12_prompt_sentiment_analysis_gemini_desktopipynb)
   - [LLama_Colab_version.ipynb](#llama_colab_versionipynb)
   - [Prompt1_Bias_Check_Visualization.ipynb](#prompt1_bias_check_visualizationipynb)
   - [Prompt1_Sentiment_Analysis_Results.csv](#prompt1_sentiment_analysis_resultscsv)
   - [Prompt2_Consistency_Check_Sentiment_Analysis_Results.csv](#prompt2_consistency_check_sentiment_analysis_resultscsv)
   - [Prompt1_Bias_Check_Sentiment_Analysis_Gemini_Desktop.ipynb](#prompt1_bias_check_sentiment_analysis_gemini_desktopipynb)
   - [Prompt2_Sentiment_Analysis_Gemini.ipynb](#prompt2_sentiment_analysis_geminiipynb)
   - [sentiment_data.csv](#sentiment_datacsv)
   - [Gemini_Sentiment_Exploration_(EDA).ipynb](#gemini_sentiment_exploration_edaipynb)
   - [Prompt1_Sentiment_Analysis_Gemini.ipynb](#prompt1_sentiment_analysis_geminiipynb)
   - [sentiment_chunkdata.csv](#sentiment_chunkdatacsv)
   - [Sentiment_Analysis_(FinBERT).ipynb](#sentiment_analysis_finbertipynb)

4. [07_User_Interface](#07_user_interface)
   - [app_with_click.py](#app_with_clickpy)
   - [app_with_click_pinecone.py](#app_with_click_pineconepy)
   - [app.py](#apppy)
