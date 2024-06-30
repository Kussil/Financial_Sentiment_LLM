import ollama

# Function to interact with Local LLAMA instance using ollama
def get_ollama_response(article, template):
    max_context_length = 8192  # Maximum context length as per the model details
    prompt = template.format(article=article)
    
    # Ensure the prompt length does not exceed the maximum context length
    if len(prompt.split()) > max_context_length:
        return f"Error: Prompt length exceeds maximum context length of {max_context_length} tokens"
    
    model = "llama3"  # Replace with the appropriate model name if needed
    response = ollama.generate(model=model, prompt=prompt)
    
    if 'response' in response:
        return response['response'].strip()
    else:
        return f"Error: {response}"
