from huggingface_hub import login
from huggingface_hub import notebook_login

def login_to_huggingface(token=None):
    if token:
        login(token=token)
    else:
        notebook_login()