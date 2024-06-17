# Import Libraries
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

# Global variables to store model, tokenizer, and llm
model_4bit = None
tokenizer = None
llm = None
template_rev_2 = "Your template here"  # Define your template here

# Define and Load Model and Tokenizer
def initialize_model():
    global model_4bit, tokenizer
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

# Create Hugging Face Pipeline
def initialize_llm():
    global llm
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

# Function to generate response
def generate_response(article):
    prompt = PromptTemplate(template=template_rev_2, input_variables=["article"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"article": article})
    return response

# Function to process the article
def processArticle(article):
    full_response = generate_response(article)
    split_response = full_response.split("</s>", 1)
    final_response = split_response[1]
    print(final_response)

# Example usage
if __name__ == "__main__":
    article = "Sample article text for testing."
    initialize_model()
    initialize_llm()
    processArticle(article)
