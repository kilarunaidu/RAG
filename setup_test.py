import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Read API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OpenAI API key not found")

print("✅ Environment configured successfully!")

try:
    client = OpenAI(api_key=api_key)

    print(".......Testing OpenAI API connection.......")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Is my setup working?"
    )

    print("✅ OpenAI API connection successful!")
    print("LLM Response:", response.output_text)

except Exception as e:
    print("❌ An error occurred:", e)
