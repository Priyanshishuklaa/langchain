from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template='Give me the name, age, city of a fictional person \n {format_instructions}',
    input_variables=['format_instructions'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt=template.format()

print(prompt)

res=model.invoke(prompt)

print(res)

final_result=parser.parse(res.content)
print(final_result)