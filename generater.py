# generate_answer.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# --- 1. Setup: Load API Key and Initialize LLM ---

# Load environment variables
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OpenAI API key not found in .env file")

# Initialize the LLM we want to use for generation
# We use ChatOpenAI for chat-based models like gpt-3.5-turbo
# 'temperature=0' makes the model more deterministic and factual
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

print("LLM initialized successfully.")

# --- 2. Create the Final Augmented Prompt ---

# Define the same prompt template as before
template_str = """
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
rag_prompt_template = PromptTemplate.from_template(template_str)

# Imagine these are our inputs from the previous steps
retrieved_context = "Vector databases like ChromaDB are essential for storing the vectorized knowledge."
user_question = "What tools are good for storing knowledge vectors?"

# Format the final prompt
final_prompt = rag_prompt_template.format(
    context=retrieved_context, 
    question=user_question
)

# --- 3. Invoke the LLM to Generate the Answer ---

print("\n--- Sending final prompt to LLM ---")
print(final_prompt)

# Invoke the LLM with the final prompt
# The LLM will "complete" the text, providing the "Answer:"
response = llm.invoke(final_prompt)

# The response object has a 'content' attribute with the answer string
final_answer = response.content

# --- 4. Display the Result ---

print("\n--- Final Generated Answer ---")
print(final_answer)