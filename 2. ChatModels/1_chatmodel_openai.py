from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
import os


load_dotenv()


# model = AzureChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
model = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version="2024-05-01-preview",
    temperature=2,
    max_tokens=10
)

res=model.invoke("what is the capital of India?")

print(res.content)