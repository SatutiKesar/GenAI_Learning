from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it")

response = llm.invoke("What is the capital of India?")
print(response.content)
