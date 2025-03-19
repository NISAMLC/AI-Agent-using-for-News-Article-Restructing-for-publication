from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
from duckduckgo_search import DDGS
from datetime import datetime

# Get current date for search
current_date = datetime.now().strftime("%Y-%m")

# Initialize LangChain with Ollama
llm = ChatOllama(
    model="gemma3:4b",  # Ensure this model is available in Ollama
    base_url="http://localhost:11434"
)


# 1. Create Internet Search Function
def get_news_articles(topic):
    print(f"Running DuckDuckGo news search for {topic}...")

    ddg_api = DDGS()
    results = ddg_api.text(f"{topic} {current_date}", max_results=5)

    if results:
        news_results = "\n\n".join([
            f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}"
            for result in results
        ])
        print(news_results)
        return news_results
    else:
        return f"Could not find news results for {topic}."


# 2. Create News Assistant
def fetch_news(topic):
    news_data = get_news_articles(topic)
    prompt = f"Summarize the following news articles into key points:\n\n{news_data}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content


# 3. Create Editor Assistant
def edit_news(news_content):
    prompt = f"Rewrite the following news articles for publication, making them engaging and structured:\n\n{news_content}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content


# 4. Run Workflow
def run_news_workflow(topic):
    print("Running news workflow...")

    # Step 1: Fetch News
    raw_news = fetch_news(topic)

    # Step 2: Edit News for Final Output
    edited_news = edit_news(raw_news)

    print("Final News Article:\n")
    print(edited_news)

    return edited_news


# Example Run
print(run_news_workflow("AI"))
