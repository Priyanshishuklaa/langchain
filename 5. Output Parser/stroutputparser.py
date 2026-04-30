from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
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

prompt1=template1.invoke({'topic':'black hole'})

result1=model.invoke(prompt1)

prompt2=template2.invoke({'text':result1.content})

result2=model.invoke(prompt2)

print(result2.content)