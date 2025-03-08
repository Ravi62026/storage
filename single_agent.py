import os
from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEARCH_TOOL = SerperDevTool(api_key=SERPER_API_KEY)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

def create_research_agent():
    return Agent (
        role="research specialist",
        goal="conduct research on the topic of the user's choice",
        backstory="You are a research specialist with a passion for finding information and sharing it with others. You are known for your attention to detail and your ability to find information quickly and accurately.",
        verbose= True,
        allow_delegation=False,
        tools=[SEARCH_TOOL],
        llm = llm,
    )
    
    
def create_research_task(agent, topic):
    return Task(
        description=f"Conduct research on the topic of {topic} and provide a summary of the information found.",
        agent=agent,
        expected_output="A summary of the information found on the topic of {topic}",
    )
    

def run_search(topic):
    agent = create_research_agent()
    task = create_research_task(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return result


if __name__ == "__main__":
    print("Welcome to the research assistant!")
    topic = input("Enter the topic you want to research: ")
    result = run_search(topic)
    print(result)
