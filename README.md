# CVX_Rice_project
This is repository for Rice University Master's student project

# Using GitHub Repositories in Google Colab

This guide explains how to link and use GitHub repositories in Google Colab, enabling you to run notebooks and access files directly from GitHub.

## Step 1: Open Google Colab

Access Google Colab by visiting [Google Colab](https://colab.research.google.com/) and start a new notebook.

## Step 2: Linking GitHub Repository

To use notebooks from GitHub:
- In your Colab notebook, navigate to `File` > `Open notebook`.
- Switch to the "GitHub" tab in the dialog that appears.
- Paste the URL of the GitHub repository or enter the username to browse repositories.
- Choose the desired notebook file (`.ipynb`) from the repository.

## Step 3: Cloning GitHub Repository

To clone an entire repository:
```python
!git clone https://github.com/username/repository.git

## Step 4: Accessing Data from the Repository
After cloning, access files using:
import pandas as pd
data = pd.read_csv('/content/repository/datafile.csv')
## Step 5: Saving Your Notebook
To save changes back to GitHub:

Choose File > Save a copy in GitHub.
Authorize Colab if prompted.
Select the target repository and branch.
Enter a commit message and click "OK".
