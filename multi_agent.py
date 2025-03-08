import os
from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEARCH_TOOL = SerperDevTool(api_key=SERPER_API_KEY)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

researcher = Agent(
    role="Senior Researcher",
    goal="uncover ground-breaking information on the {topic} of the user's choice",
    verbose=True,
    memory=True,
    backstory=(
    """
        Driven by curiosity, you are at the forefront of innovation, eager to explore and share knowledge that could change the world.
    """
    ),
    tools=[SEARCH_TOOL],
    allow_delegation=True
)

writer = Agent(
    role="Senior Writer",
    goal="write a comprehensive report on the {topic} of the user's choice",
    verbose=True,
    memory=True,
    backstory=(
        """
        You are a seasoned writer with a knack for crafting engaging and informative reports.
        """
    ),
    tools=[SEARCH_TOOL],
    allow_delegation=True
)

research_task = Task(
    description=(
        "Identity the next big trend in {topic}"
        "Focus on identifying pros and cons and the overall narrative"
        "Your final reposrt should clearly articulate the key points"
        "Its market opportunities, and potential riskd"
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    tools=[SEARCH_TOOL],
    agent=researcher,
)

writer_task = Task(
    description=(
        "Write a comprehensive report on the {topic} of the user's choice"
        "Use the information from the researcher to craft a detailed and engaging report"
        "Ensure the report is well-structured and easy to understand"
        "Include a clear conclusion and recommendations"
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    tools=[SEARCH_TOOL],
    agent=writer,
    output_file="report.md",
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    process=Process.sequential,
)

results = crew.kickoff(inputs={"topic": "AI in indian legal and judicial system, and also check for chain verdict, which try to use ai and blockchain to make the system more efficient"})

print(results)