# This is the same code as above but using the google genai api instead of huggingface endpoint because the huggingface endpoint is not working for me.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional character person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()} #it does not take output from user, it takes the format instruction from the parser and adds it to the prompt template. And it fills before running the model. So the model will get the prompt with the format instruction and it will know how to format the output.
)

chain = template | model | parser

result = chain.invoke({})

print(result['name'])
print(type(result))