# Flat_Stock_Data

This folder contains scripts and models for flat stock data modeling. Note that 02_Prompt2_Gemini_Random_Forest_LabelEncoding.ipynb is the final notebook used to create the output csv used in our app.

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
- `02_Prompt2_Gemini_Data_Prep.ipynb`: Data preparation for Gemini modeling (Prompt 2).
- `02_Prompt2_Gemini_Logistic_Regression.ipynb`: Logistic Regression model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_MLPClassifier.ipynb`: MLP Classifier model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_Prepped_Stock_Data.csv`: Prepped stock data for Gemini models (Prompt 2).
- `02_Prompt2_Gemini_Random_Forest_LabelEncoding.ipynb`: Random Forest model with label encoding using Gemini (Prompt 2). NOTE: This is the final version used in our app.
- `02_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using one hot encoding Gemini (Prompt 2).
- `02_Prompt2_Gemini_SVM.ipynb`: SVM model using Gemini (Prompt 2).
- `02_Prompt2_Gemini_XGBoost.ipynb`: XGBoost model using Gemini (Prompt 2).
- `gemini_sentiment_predictions.csv`: CSV file containing sentiment predictions for Gemini data.  NOTE: This file contains the results which will be used in our app.

#### Source_Testing_Folder
- `EC_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using only Earnings Call sentiment data (Prompt 2).
- `IR_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using only Investment Research sentiment data (Prompt 2).
- `ProQuest_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using ProQuest news article sentiment data (Prompt 2).
- `SEC_Prompt2_Gemini_Random_Forest.ipynb`: Random Forest model using SEC sentiment data (Prompt 2).