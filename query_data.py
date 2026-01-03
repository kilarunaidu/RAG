import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- 1. Setup: Connect to the Existing Database ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found in .env file")

# Define the same embedding model used during indexing
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to the persistent directory
# The wrapper handles the connection and the embedding logic for you
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model,
    collection_name="my_knowledge_base"
)

print(f"✅ Connected to the collection.")

# --- 2. Define the Query and Perform Retrieval ---
user_query = "What is the best way to store vector knowledge?"

# Perform similarity search
# This returns a list of LangChain Document objects
results = vector_db.similarity_search(user_query, k=2)

# --- 3. Inspect the Results ---
print(f"\nQuery: '{user_query}'")
print("\n--- Top Results ---")

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Document: {doc.page_content}")
    # You can also access metadata if you stored any
    print(f"  Metadata: {doc.metadata}")
    print("-" * 20)