import os
from dotenv import load_dotenv
# Updated Import Paths
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Verify API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Ensure it is set in your .env file.")

# 1. Sample text
sample_text = """Welcome to the RAG Guided Path. This course covers everything from the basics to advanced techniques.
The first major section is on Data Preparation. It includes loading, cleaning, and chunking documents.
Final section: evaluation."""

# 2. Text Splitting
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20
)
chunks = recursive_splitter.split_text(sample_text)

print(f"Split into {len(chunks)} chunks.")

# 3. Instantiate the model 
# (This requires the langchain-openai package)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. Vectorize
try:
    vector_embeddings = embeddings_model.embed_documents(chunks)
    print(f"Successfully vectorized {len(vector_embeddings)} chunks.")
    print(f"Vector dimensions: {len(vector_embeddings[0])}")
    print(f"Preview: {vector_embeddings[0][:5]}")
except Exception as e:
    print(f"An error occurred during vectorization: {e}")