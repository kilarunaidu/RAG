# end_to_end_rag.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Setup: Load DB, LLM, and Retriever ---
print("Setting up the RAG chain...")

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OpenAI API key not found in .env file")

# Load our existing vector store
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=OpenAIEmbeddings()
)

# Initialize the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define our prompt template
template_str = """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know.

Context: {context}
Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template_str)

# --- 2. Define the RAG Chain using LCEL ---

# This helper function formats the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This is where the magic happens!
# We define the sequence of operations using the pipe operator.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain created successfully.")

# --- 3. Invoke the Chain and Get an Answer ---

user_question = "How are RAG systems helpful?"

print(f"\nInvoking chain with question: '{user_question}'")

# The invoke method takes the user's question and runs the entire pipeline
final_answer = rag_chain.invoke(user_question)

print("\n--- Final Answer from RAG Chain ---")
print(final_answer)