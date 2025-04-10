# llm.py
from langchain_ollama import OllamaLLM 
import config

# Create an instance of the LangChain Ollama LLM.
# Note: You can pass stream=False to disable streaming.
ollama_llm = OllamaLLM(
    model=config.OLLAMA_MODEL,
    base_url="http://localhost:11434"
)