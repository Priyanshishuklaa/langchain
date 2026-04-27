from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

embeddings=AzureOpenAIEmbeddings(model='text-embedding-ada-002')

result=embeddings.embed_query("Delhi is the capital of India")
print(str(result))


# import os
# from langchain_openai import AzureOpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv(override=True)

# print("KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
# print("ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
# print("DEPLOYMENT:", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))

# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION")
# )

# result = embeddings.embed_query("Delhi is the capital of India")
# print(result)