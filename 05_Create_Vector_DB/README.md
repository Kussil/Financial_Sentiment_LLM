# 05_Create_Vector_DB Folder

This folder contains notebooks to process the sentiments from FinBERT and Gemini, and to embed text of each article as vectors with associated date, ticker, and sentiment meta data.

## Directory Contents

### Jupyter Notebooks
- **FinBERT**
  - **Create_Vector_Database_FinBERT.ipynb**
- **Gemini**
  - **Create_Vector_Database_Gemini.ipynb**

The Jupyter Notebook handles embedding the text as vectors and utilizes Pinecone API to write a vector database. FinBERT and Gemini are written into separate vector databases.

### CSV Files
- **Gemini**
  - **Article_Chunk_Reference_pt1.csv**
  - **Article_Chunk_Reference_pt1.csv**
  - **Article_Chunk_Reference_pt1.csv**

For Gemini the CSV files are used as reference for the vector database for after the vector similarity matching in order to show the associated article text chunk.