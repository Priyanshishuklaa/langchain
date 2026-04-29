from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv(override=True)

model=AzureChatOpenAI(model='gpt-4.1-mini',api_version='2024-05-01-preview') #gpt-4.1-mini or gpt-4o

chat_history=[
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input=input("You:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower()=='exit':
        print("Exiting the chatbot. Goodbye!")
        break
    res=model.invoke(chat_history)
    chat_history.append(AIMessage(content=res.content))
    print("AI:",res.content)
print(chat_history)