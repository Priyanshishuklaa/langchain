from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

model=AzureChatOpenAI(model='gpt-4.1-mini',api_version='2024-05-01-preview') #gpt-4.1-mini or gpt-4o

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain?")
]

res=model.invoke(messages)
messages.append(AIMessage(content=res.content))
print(messages)