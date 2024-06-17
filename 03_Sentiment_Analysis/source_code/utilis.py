def initialize_model():

# Define Quantization
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)
# Load Model and Tokenizer
    model_4bit = AutoModelForCausalLM.from_pretrained( "meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto",quantization_config=quantization_config, )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

initialize_model()

def initialize_llm():
    # Create Hugging Face Pipeline
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

def generate_response(article):
  prompt = PromptTemplate(template=template_rev_2, input_variables=["article"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run({"article":article})
  return response

def processArticle():
    # Test Function
    full_response = generate_response(article)
    split_response = full_response.split("</s>", 1)
    final_response = split_response[1]
    print(final_response)

processArticle()

