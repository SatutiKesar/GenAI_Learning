from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel

load_dotenv()



prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a LinkedIn post about {topic}',
    input_variables=['topic']
)


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, model, parser),
    'linkedin':RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'QA RNN based pytorch project'})

print(result['tweet'])
print(result['linkedin'])