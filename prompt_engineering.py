import os
from dotenv import load_dotenv

# Modern LangChain imports (2026 standard)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# 2. Define the LLM and Embeddings
# Using gpt-4o-mini and text-embedding-3-small for cost-efficiency
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Connect to your existing Vector Store
# (Assuming you have already created 'chroma_db' in previous steps)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_knowledge_base"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Define the Prompt Template
# We use ChatPromptTemplate for better performance with ChatModels
template = """
You are a helpful and precise assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""
rag_prompt = ChatPromptTemplate.from_template(template)

# 5. Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Construct the RAG Chain
# This 'pipes' the components together: Context -> Prompt -> LLM -> String
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Test the Full LLM ---
user_question = "What is the best way to store vector knowledge?"

print(f"Asking: {user_question}...")
response = rag_chain.invoke(user_question)

print("\n--- AI Response ---")
print(response)