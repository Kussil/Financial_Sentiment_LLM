from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model_and_tokenizer(model_name, quantization_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def create_pipeline(model, tokenizer):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=5000,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
