
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview")


# if __name__ == "__main__":
#     response = llm.invoke("Two most important ingradient in samosa are ")
#     print(response.content)


from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_llm():
    """Initialize the LLM with the Groq API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Environment variable 'GROQ_API_KEY' is not set.")
    return ChatGroq(groq_api_key=api_key, model_name="llama-3.2-90b-text-preview")

def ask_groq(prompt):
    """Send a prompt to the Groq API and return the response."""
    if not prompt.strip():
        return "Prompt cannot be empty. Please enter a valid question."
    try:
        llm = initialize_llm()
        response = llm.invoke(prompt)
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Response: {response.content}")
        return response.content
    except ValueError as ve:
        logging.error(f"Configuration Error: {ve}")
        return "Configuration error. Please check your API key."
    except Exception as e:
        logging.error(f"Error invoking Groq API: {e}")
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    question = input("Enter your question (or press Enter for a default prompt): ")
    if not question.strip():
        question = "What is the importance of AI in modern technology?"
    print(f"Response: {ask_groq(question)}")
