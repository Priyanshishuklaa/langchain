from langchain_openai import AzureChatOpenAI

from typing import TypedDict, Annotated, Optional, Literal
import os


  
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


model=AzureChatOpenAI(azure_deployment=AZURE_OPENAI_DEPLOYMENT, api_version=AZURE_OPENAI_API_VERSION)

#schema


s=model.with_structured_output(Review)

res=s.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.")

print(res)
print('--------------------------------------')
print(res['summary'])
print('--------------------------------------')
print(res['sentiment'])