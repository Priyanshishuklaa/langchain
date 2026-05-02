from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv() 

llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name="fact1", description="Fact1 about the topic"),
    ResponseSchema(name="fact2", description="Fact2 about the topic"),
    ResponseSchema(name="fact3", description="Fact3 about the topic")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template='Give me 3 facts about {topic} in the following format: \n {format_instructions}',
    input_variables=['topic'],  
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt=template.invoke({'topic':'black hole'})
res=model.invoke(prompt)
final_result=parser.parse(res.content)  
print(final_result)


