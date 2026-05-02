from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")

model=ChatHuggingFace(llm=llm)
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18,description="Age of the person")
    city: str = Field(description="City where the person lives")

parser=PydanticOutputParser(pydantic_object=Person)

# template=PromptTemplate(
#     template='Generate the name, age and city of a fictional {place} person \n {format_instructions}',
#     input_variables=['place'],
#     partial_variables={'format_instructions': parser.get_format_instructions()}

# )

template = PromptTemplate(
    template="""
Generate the name, age and city of a fictional {place} person.

IMPORTANT:
- Return ONLY valid JSON
- Do NOT include explanation or code
- Do NOT include ``` or text

{format_instructions}
""",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


chain=template | model | parser

res=chain.invoke({'place':'Indian'})    
print(res)