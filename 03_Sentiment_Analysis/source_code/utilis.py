# Import Libraries
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

# Global variables to store model, tokenizer, and llm
model_4bit = None
tokenizer = None
llm = None
template_rev_2 = """<s>Summarize the following article and classify it into the categories with sentiment (Positive, Neutral, Negative, N/A if not applicable). Use the format provided in the example.

Categories:
- Finance: Financial performance, earnings, investments
- Production: Oil and gas production levels and outputs
- Reserves / Exploration / Acquisitions / Mergers / Divestments: Activities related to reserves, exploration, acquisitions, mergers, divestments
- Environment / Regulatory / Geopolitics: Environmental impact, regulations, geopolitical issues
- Alternative Energy / Lower Carbon: Initiatives in alternative energy and lower carbon emissions
- Oil Price / Natural Gas Price / Gasoline Price: Pricing of oil, natural gas, gasoline

Example:
Article: "The company reported a 20% increase in earnings, driven by higher oil prices and increased production levels. They also announced a new exploration project in the North Sea."

Output:
- Finance == Positive
- Production == Positive
- Reserves / Exploration / Acquisitions / Mergers / Divestments == Positive
- Environment / Regulatory / Geopolitics == Neutral
- Alternative Energy / Lower Carbon == N/A
- Oil Price / Natural Gas Price / Gasoline Price == Positive

Article: {article}

Your Output:
- Finance == [Sentiment]
- Production == [Sentiment]
- Reserves / Exploration / Acquisitions / Mergers / Divestments == [Sentiment]
- Environment / Regulatory / Geopolitics == [Sentiment]
- Alternative Energy / Lower Carbon == [Sentiment]
- Oil Price / Natural Gas Price / Gasoline Price == [Sentiment]
</s>

""" 

# Define and Load Model and Tokenizer
def initialize_model():
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
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
