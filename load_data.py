from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader



loader = TextLoader("./data/my_notes.txt")
pdf_loader = PyPDFLoader("./data/attention_all_you_need.pdf")
pages = pdf_loader.load_and_split()

docs = loader.load()

print(f"Loaded {len(docs)} documents(s).")

print("\n--- Here's the content: ---")
print(docs[0].page_content)
print("\n--- And here's the metadata: ---")
print(docs[0].metadata)

print("\n--- Content from page 5: ---")
print(pages[4].page_content[:400]  )
print(pages[4].metadata)


from langchain_text_splitters import CharacterTextSplitter


# Some text to split
long_text = (
    "Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing LLM performance. "
    "It combines a retriever, which pulls relevant documents, with a generator LLM. "
    "This process grounds the model in facts, reducing hallucinations. The choice of chunking "
    "strategy is critical for retrieval quality. We will explore several methods."
)

# 1. Initialize the splitter
text_splitter = CharacterTextSplitter(
    separator="\n",  # We'll keep this simple for now
    chunk_size=120,  # Max characters per chunk
    chunk_overlap=20, # Characters to overlap between chunks
    length_function=len,
)

# 2. Split the text
chunks = text_splitter.split_text(long_text)

# 3. Inspect the chunks
print(f"Original text length: {len(long_text)}")
print(f"Split into {len(chunks)} chunks.")
print("-" * 20)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (length: {len(chunk)}):\n\"{chunk}\"")
    print("-" * 20)