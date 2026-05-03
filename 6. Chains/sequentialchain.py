from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv(override=True)

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",   
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate a 5 pointer summary from the following input text: {text}",
    input_variables=["text"]
)

model=AzureChatOpenAI(azure_deployment="gpt-4.1-mini",  api_version="2024-05-01-preview")

parser= StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

res=chain.invoke({'topic':'LangChain significance in AI development'})

print(res)

print(chain.get_graph().print_ascii())