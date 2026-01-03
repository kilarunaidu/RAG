from langchain_text_splitters import CharacterTextSplitter

long_text = (
    "Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing LLM performance. "
    "It combines a retriever, which pulls relevant documents, with a generator LLM. "
    "This process grounds the model in facts, reducing hallucinations. The choice of chunking "
    "strategy is critical for retrieval quality. We will explore several methods."
)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=120,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_text(long_text)

print(f"Original text length: {len(long_text)}")
print(f"Split into {len(chunks)} chunks.")
print("-" * 20)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (length: {len(chunk)}):\n\"{chunk}\"")
    print("-" * 20)
