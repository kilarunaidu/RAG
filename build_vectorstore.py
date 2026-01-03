import os
from dotenv import load_dotenv
# Updated Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma 

# --- 1. Load and Prepare ---
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OpenAI API key not found in .env file")

sample_data = """
RAG is a powerful AI technique. It combines retrieval with generation.
Vector databases like ChromaDB are essential for storing the vectorized knowledge.
The key steps are loading, chunking, embedding, and storing.
This process allows LLMs to access external, up-to-date information.
"""

with open("./data/sample_doc.txt", "w") as f:
    f.write(sample_data)

loader = TextLoader("./data/sample_doc.txt")
documents = loader.load()

# Using 150/20 split as requested
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")

# --- 2. Initialize Embeddings & Vector Store ---
# This automatically handles the embedding of your chunks
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Instead of using chromadb.PersistentClient manually, 
# the LangChain Chroma wrapper handles IDs and persistence for you.
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="./chroma_db",
    collection_name="my_knowledge_base"
)

print(f"Successfully saved chunks to ChromaDB.")


from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings_model)

print("✅ FAISS in-memory database created successfully.")