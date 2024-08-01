# 07_User_Interface Folder

This folder contains scripts to implement our Retreival Augmented Generation (RAG) based application to derive insights about stock price movement from relevant articles.

## Directory Contents
- **app_demo.py**


The app_demo.py is the locally run app which can be launch via your local terminal. The streamlit_io_app.py is the versions of the app that has been loaded to the Azure environment.

## Running the Application
To run the Streamlit application, execute the following command in your terminal:

```sh
streamlit run app_demo.py
```
This command will start a local web server, and you can view the application by navigating to http://localhost:8501 in your web browser.

## How to Use
1. Ensure that you have GOOGLE_API_KEY for Gemini set as an environment variable. To get an API key you can go to: <https://ai.google.dev/gemini-api/docs/api-key>.
2. Select a date from the graph by clicking or using the dropdown select box. This will display the stock price change for that day.
3. The sentiments for the last 7 days will be summarized in a chart showing the sentiments for each article for each sentiment category.
4. Click the Generate Response button to get a list of insights to the stock price change based on articles from the last 7 days.


## Deployment Summary
We containerized the web application by creating a Docker image that includes only the essential files and dependencies required to run the app, ensuring a minimal and efficient container. This container was built using Docker Buildx to guarantee compatibility with Azure's AMD architecture. The Docker image was then tagged and pushed to Azure Container Registry (ACR) for easy management and deployment. We configured the Azure Web App to pull and use this Docker image directly from the ACR using Azure CLI commands, ensuring that the updated and fixed version of the application is deployed and running smoothly in the Azure environment. This work was done on the https://github.com/Kussil/Financial_Sentiment_LLM.git branch, which includes all recent bug fixes and updates necessary for the deployment.