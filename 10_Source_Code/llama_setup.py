import ollama

# Function to interact with Local LLAMA instance using ollama
def get_ollama_response(article, template):
    prompt = template.format(article=article)
    model = "llama3"  # Replace with the appropriate model name if needed
    response = ollama.generate(model=model, prompt=prompt)
    
    if 'response' in response:
        return response['response'].strip()
    else:
        return f"Error: {response}"
