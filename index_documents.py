import os
from dotenv import load_dotenv
# Corrected modern imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma  # The specialized integration package

# --- 1. Prepare Your Data ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Ensure the file exists for this example
os.makedirs("./data", exist_ok=True)
if not os.path.exists("./data/company_handbook.txt"):
    with open("./data/company_handbook.txt", "w") as f:
        f.write("Welcome to the company. Our policy is 'Safety First'. RAG is our primary AI tool.")

loader = TextLoader("./data/company_handbook.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Prepared {len(chunks)} chunks from the document.")

# --- 2. Connect and Index ---
# Using the current standard '3-small' model (cheaper and faster)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Using Chroma.from_documents is the 'gold standard' because:
# 1. It handles the ID generation automatically
# 2. It correctly maps LangChain metadata to Chroma
# 3. It bridges the embedding function mismatch for you
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="./my_rag_db",
    collection_name="employee_handbook"
)

print(f"✅ Successfully loaded and indexed {len(chunks)} documents!")

# --- 3. Optional: Test a Search ---
query = "What is our primary AI tool?"
results = vector_db.similarity_search(query, k=1)
print(f"Search Result: {results[0].page_content}")