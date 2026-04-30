from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# llm=HuggingFaceEndpoint(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

# model=ChatHuggingFace(llm=llm)

model=AzureChatOpenAI(azure_deployment="gpt-4.1-mini", api_version="2024-05-01-preview")

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'black hole'})
print(result)
