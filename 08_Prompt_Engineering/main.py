from huggingface_login import login_to_huggingface
from config import get_quantization_config
from model import load_model_and_tokenizer, create_pipeline
from llm_chain import initialize_llm, generate_response
from utils import process_response
from templates import template_rev_2

def main():
    token = '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'
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Login
    login_to_huggingface(token)

    # Get Quantization Config
    quant_config = get_quantization_config()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, quant_config)

    # Create Pipeline
    pipeline_inst = create_pipeline(model, tokenizer)

    # Initialize LLM
    llm = initialize_llm(pipeline_inst)

    # Article for Testing
    article = """
    Stock Report | May 16, 2024 | NYSESymbol: MRO | MRO is in the S&P 500
    Marathon Oil Corporation
    Recommendation Price 12-Mo. Target Price Report Currency Investment Style
    HOLD Â« Â« Â« Â« Â« USD 26.44 (as of market close May 15, 2024) USD 28.00 USD Mid-Cap Blend
    ...
    """

    # Generate Response
    full_response = generate_response(article, template_rev_2, llm)

    # Process and Print Final Response
    final_response = process_response(full_response)
    print(final_response)

if __name__ == "__main__":
    main()
