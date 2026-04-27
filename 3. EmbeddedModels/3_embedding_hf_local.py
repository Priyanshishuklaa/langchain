from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# text='Delhi is the capital of India'

# result=embedding.embed_query(text)
# print(result)

documents=['Delhi is the capital of India',
           'Mumbai is the financial capital of India',
           'Bangalore is the IT hub of India']

res=embedding.embed_documents(documents)
print(res)