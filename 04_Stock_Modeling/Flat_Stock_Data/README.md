# Flat_Stock_Data

This folder contains scripts and models for flat stock data modeling.

## Contents

### FinBERT
- `01_FinBERT_Data_Prep.ipynb`: Data preparation for FinBERT modeling.
- `01_FinBERT_Logistic_Regression.ipynb`: Logistic Regression model using FinBERT.
- `01_FinBERT_MLPClassifier.ipynb`: MLP Classifier model using FinBERT.
- `01_FinBERT_Prepped_Stock_Data.csv`: Prepped stock data for FinBERT models.
- `01_FinBERT_Random_Forest.ipynb`: Random Forest model using FinBERT.
- `01_FinBERT_SVM.ipynb`: SVM model using FinBERT.
- `01_FinBERT_XGBoost.ipynb`: XGBoost model using FinBERT.

### Gemini_Prompt1
- `02_Prompt1_Gemini_Data_Prep.ipynb`: Data preparation for Gemini modeling (Prompt 1).
- `02_Prompt1_Gemini_Logistic_Regression.ipynb`: Logistic Regression model using Gemini (Prompt 1).
- `02_Prompt1_Gemini_MLPClassifier.ipynb`: MLP Classifier model using Gemini (Prompt 1).
- `02_Prompt1_Gemini_Prepped_Stock_Data.csv`: Prepped stock data for Gemini models (Prompt 1).
- `02_Prompt1_Gemini_Random_Forest.ipynb`: Random Forest model using Gemini (Prompt 1).
- `02_Prompt1_Gemini_SVM.ipynb`: SVM model using Gemini (Prompt 1).
- `02_Prompt1_Gemini_XGBoost.ipynb`: XGBoost model using Gemini (Prompt 1).

### Gemini_Prompt2
#### Source_Testing
- `EC_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using EC data (Prompt 2).
- `IR_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using IR data (Prompt 2).
- `ProQuest_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using ProQuest data (Prompt 2).
- `SEC_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using SEC data (Prompt 2).

#### Main Notebooks
- `02_Prompt2_Gemini_Data_Prep.ipynb`: Data preparation for Gemini modeling (Prompt 2).
- `02_Prompt2_Gemini_Logistic_Regression.ipynb`: Logistic Regression model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_MLPClassifier.ipynb`: MLP Classifier model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_Prepped_Stock_Data.csv`: Prepped stock data for Gemini models (Prompt 2).
- `02_Prompt2_Gemini_Random_Forest_LabelEncoding.ipynb`: Random Forest model with label encoding using Gemini (Prompt 2).
- `02_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_SVM.ipynb`: SVM model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_XGBoost.ipynb`: XGBoost model using Gemini (Prompt 2).
- `gemini_sentiment_predictions.csv`: CSV file containing sentiment predictions for Gemini data.

## Prerequisites

To use the notebooks in this directory, ensure you have the following:

1. **Gemini API key** - Obtain a key to access the Gemini API. Follow the instructions [here](https://www.gemini.com/cryptopedia/api).
2. **Hugging Face API key** - Obtain a key to access Hugging Face models. Follow the instructions [here](https://huggingface.co/docs/api-inference/quicktour#getting-started).
3. **GitHub API key** - Obtain a key to access GitHub API. Follow the instructions [here](https://docs.github.com/en/rest/overview/other-authentication-methods#personal-access-tokens).
4. **Google Colab access** - Notebooks are designed to work using Google Colab VM machines with GPU. You can access Google Colab [here](https://colab.research.google.com/).

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/04_Stock_Modeling.git
