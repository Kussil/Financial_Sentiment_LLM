# utils.py

# Necessary imports
from huggingface_hub import notebook_login
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def login_to_huggingface():
    notebook_login()

def get_huggingface_pipeline(model_id, task):
    pipe = pipeline(task=task, model=model_id, device=0)
    return HuggingFacePipeline(pipeline=pipe)
