from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style and leadership skills.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing abilities.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and has been a key player for the Indian cricket team.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorker deliveries."
]

query = "Tell me about jasprit bumrah"

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score:", score)
