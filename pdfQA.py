import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Setup: Load Environment and Define the Chain ---

# Load environment variables
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OpenAI API key not found in .env file")

# This helper function will build our RAG chain
def build_rag_chain(pdf_path):
    print("Building RAG chain for PDF...")

    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create an in-memory vector store
    # For a real app, you'd use a persistent directory
    vector_store = Chroma.from_documents(chunks, OpenAIEmbeddings())

    # Create the retriever
    retriever = vector_store.as_retriever()

    # Define the prompt template
    template = """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Format docs helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain built successfully!")
    return rag_chain

# --- 2. Main Interaction Loop ---

def main():
    pdf_path = "./data/attention_all_you_need.pdf" # Make sure this path is correct
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    # Build the chain once when the script starts
    rag_bot = build_rag_chain(pdf_path)

    print("\nPDF Q&A Bot is ready! Ask a question about the document.")
    print("Type 'quit' to exit.")

    while True:
        user_question = input("\nYour Question: ")
        if user_question.lower() == 'quit':
            break
        
        # Get the answer from the RAG chain
        answer = rag_bot.invoke(user_question)
        print("\nBot's Answer:", answer)

if __name__ == "__main__":
    main()