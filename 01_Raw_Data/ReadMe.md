# 01_Raw_Data Folder

This folder contains the raw data which was collected through data wrangling and data scraping following the selection of sources that were used. The four source types each have a subdirectory folder which structure is described below. There is also a subdirectory which contains example notebooks of the processes that were utilized to scrape the data for certain source types. This data was compiled to .csv format for further cleaning which can be seen in the 02_Cleaned_Data directory.

## Directory Contents
- **SEC_Filings.csv**
  This .csv file contains the formatted and extracted text from the SEC HTML data source.
### Subdirectories (Sources)
- **Earnings_Transcripts**
  Contains the formatted .csv files with extracted text from Earnings Call Transcripts.
- **Investment_Research_Analysts_Reports**
- Contains the formatted .csv files with extracted text of various Analyst Reports from Investment Research.
- **ProQuest_News_Articles**
  Contains the formatted .csv files with extracted text of various News Articles from ProQuest.
- **SEC_Data**
  Contains the HTML files for the various SEC Data (8-k, 10-k, etc.)
### Subdirectories (References)
- **Extraction_Notebooks**
  Contains the notebooks which were used to extract text and save to .csv format from all sources.
  Note: Various sources were manually queried and PDF's saved locally, or search results with URL's saved locally.
