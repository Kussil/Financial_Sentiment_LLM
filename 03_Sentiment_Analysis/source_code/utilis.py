# Import Libraries
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.schema.runnable import RunnableSequence
import torch

# Global variables to store model, tokenizer, and llm
model_4bit = None
tokenizer = None
llm = None

# Prompt template for sentiment analysis
template_rev_2 = """<s>Given the text, analyze the content and perform sentiment analysis across multiple predefined categories.

Sentiment options:
  - Positive
  - Neutral
  - Negative

Categories:
  - Finance
  - Production
  - Reserves / Exploration / Acquisitions / Mergers / Divestments
  - Environment / Regulatory / Geopolitics
  - Alternative Energy / Lower Carbon
  - Oil Price / Natural Gas Price / Gasoline Price

Each category should be evaluated and given a sentiment output derived from the text.
If a category is not mentioned or relevant based on the text content, mark it as 'Neutral'.

The text is below:
{article_text}

Remember to summarize your final answers in the following format exactly:
- Category - Sentiment
- Category - Sentiment
- Category - Sentiment
- Category - Sentiment
- Category - Sentiment
- Category - Sentiment

Make sure to use plain text and stick to the given categories and sentiment options.
DO NOT bold or bullet the output summary.
"""

# Define and Load Model and Tokenizer
def initialize_model():
    """Initializes the model and tokenizer."""
    global model_4bit, tokenizer
    if model_4bit is None or tokenizer is None:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
            model_4bit = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                device_map="auto",
                quantization_config=quantization_config)
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        except Exception as e:
            print(f"Error initializing model or tokenizer: {e}")

# Create Hugging Face Pipeline
def initialize_llm():
    """Initializes the Hugging Face pipeline."""
    global llm
    if llm is None:
        try:
            pipeline_inst = pipeline(
                "text-generation",
                model=model_4bit,
                tokenizer=tokenizer,
                use_cache=True,
                device_map="auto",
                max_length=5000,
                do_sample=True,
                top_k=5,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id)
            llm = HuggingFacePipeline(pipeline=pipeline_inst)
        except Exception as e:
            print(f"Error initializing LLM pipeline: {e}")

# Function to generate response
def generate_response(article_text):
    """Generates the response for the given article text using the LLM."""
    prompt = PromptTemplate(template=template_rev_2, input_variables=["article_text"])
    sequence = RunnableSequence([prompt, llm])
    response = sequence.invoke({"article_text": article_text})
    return response['generated_text']

# Function to process the article
def process_article(article_text):
    """Processes the article text and returns the sentiment analysis result."""
    full_response = generate_response(article_text)
    split_response = full_response.split("</s>", 1)
    if len(split_response) > 1:
        final_response = split_response[1]
    else:
        final_response = "Error processing response"
    return final_response



