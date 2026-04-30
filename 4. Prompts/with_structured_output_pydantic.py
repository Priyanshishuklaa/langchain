from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os
load_dotenv(override=True)

  
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


model=AzureChatOpenAI(azure_deployment=AZURE_OPENAI_DEPLOYMENT, api_version=AZURE_OPENAI_API_VERSION)

#schema
class Review(BaseModel):
    key_themes: list[str]=Field(description="Write down all the key themes discussed in the review in a list")
    summary: str=Field(description="A brief summary of the review")
    sentiment: Literal['pos','neg']=Field(description="Return sentiment of the review, either 'pos' or 'neg'")
    pros: Optional[list[str]]=Field(description="Write down all the pros inside a list")
    cons: Optional[list[str]]=Field(description="Write down all the cons inside a list")
    name: Optional[str]=Field(description="Write the name of the reviewer ")

s=model.with_structured_output(Review)

res=s.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this. Reviewed by Priyanshi")
print('--------------------------------------')
print(res)
print('--------------------------------------')
print(res.key_themes)
print('--------------------------------------')
print(res.sentiment)