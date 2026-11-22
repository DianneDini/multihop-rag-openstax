import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="This is a test for my RAG thesis."
)

embedding = response.data[0].embedding

print("BERHASIL!")
print("Jumlah dimensi:", len(embedding))
print("Token digunakan:", response.usage.total_tokens)