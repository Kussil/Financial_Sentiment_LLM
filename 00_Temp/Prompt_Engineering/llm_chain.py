from langchain import PromptTemplate, LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

def initialize_llm(pipeline_inst):
    return HuggingFacePipeline(pipeline=pipeline_inst)

def generate_response(article, template, llm):
    prompt = PromptTemplate(template=template, input_variables=["article"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"article": article})
    return response
