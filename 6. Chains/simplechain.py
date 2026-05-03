from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

prompt=PromptTemplate(
    template="Generate 5 line Description about {topic}",
    input_variables=["topic"]
)

model=AzureChatOpenAI(azure_deployment="gpt-4.1-mini",  api_version="2024-05-01-preview")

parser=StrOutputParser()

chain =prompt | model | parser
res=chain.invoke({'topic':'Python Programming'})
print(res)

chain.get_graph().print_ascii()
