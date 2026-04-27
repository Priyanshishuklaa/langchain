from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

model=AzureChatOpenAI(model='gpt-4.1-mini',api_version='2024-05-01-preview') #gpt-4.1-mini or gpt-4o

while True:
    user_input=input("You:")
    if user_input.lower()=='exit':
        print("Exiting the chatbot. Goodbye!")
        break
    res=model.invoke(user_input)
    print("AI:",res.content)