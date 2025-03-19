from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

# Initialize LangChain Ollama model
llm = ChatOllama(
    model="gemma3:4b",  # Ensure model name is correct
    base_url="http://localhost:11434"  # Connect to Ollama
)

# Send a test prompt
messages = [HumanMessage(content="Create a meal plan for a week.")]
response = llm.invoke(messages)

print(response.content)