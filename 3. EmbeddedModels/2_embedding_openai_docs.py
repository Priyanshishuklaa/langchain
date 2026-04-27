from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

embeddings=AzureOpenAIEmbeddings(model='text-embedding-ada-002')

documents=["Delhi is the capital of India",
           "Mumbai is the financial capital of India",
           "Bangalore is the IT hub of India"]

result=embeddings.embed_documents(documents)
print(str(result))
