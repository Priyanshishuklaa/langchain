from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')


result=model.invoke("what is the capital of India?")
print(result.content)


# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# for model in genai.list_models():
#     print(model.name, model.supported_generation_methods)