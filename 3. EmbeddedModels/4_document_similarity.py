from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings=AzureOpenAIEmbeddings(model='text-embedding-ada-002')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about MS dhoni'

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0] #2d

index, score= sorted(list(enumerate(similarities)),key=lambda x:x[1])[-1]

print(f"Most similar document: {documents[index]} with similarity score: {score}")


