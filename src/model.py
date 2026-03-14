from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
qwen = ChatOllama(model="qwen2.5:3b", temperature=0)