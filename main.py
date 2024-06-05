"""
Name
    main.py

Author
    Written by Rip&Tear - CrewAI Discord Moderator .riptear
    
Date Sat 13th Apr 2024
    
Description
    This is a basic example of how to use the CrewAI library to create a simple research task. 
    The task is to research the topic of "70s and 80s British rock bands" and provide 5 paragraphs of information on the topic. 
    The task is assigned to a single agent (Researcher) who will use the ChatOllama model to generate the information. 
    The result of the task is written to a file called "research_result.txt".

Usage
    python main.py
    
Output
    The output of the task is written to a file called "research_result.txt"."""

# Import required libraries - make sure the crewai and langchain_community packages are installed via pip
import os
from langtrace_python_sdk import langtrace
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Access the environment variables
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
LANGTRACE_API_KEY = os.getenv("LANGTRACE_API_KEY")
print(OPENAI_API_BASE)
print(OPENAI_API_KEY)
print(OPENAI_MODEL_NAME)
print(LANGTRACE_API_KEY)
langtrace.init(api_key=LANGTRACE_API_KEY)
from crewai import Agent, Crew, Process, Task
from tools import ExaSearchToolset
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME, base_url=OPENAI_API_BASE, openai_api_key="OPENAI_API_KEY"
)


# Create a function to log to a file with the date as the filename - this will be used as a callback function for the agent. this could be as complex as you like
def write_result_to_file(result):
    filename = "raw_output.log"
    with open(filename, "a") as file:
        file.write(str(result))


# Define post-process functions
def store_in_context(output, context):
    print("researcher post_processor: Storing in context:", output)
    context["research_material"] = output
    print("Updated context:", context)
    return output


def check_context_and_summarize(inputs, context):
    print("evaluator post_processor:", inputs)
    print("Current context:", context)
    research_material = context.get("research_material", None)
    if research_material:
        summary = f"Summary of the research material: {research_material[:100]}..."
        print("Context found and summarized:", summary)
        return summary
    else:
        print("Research material not found in context.")
        return "Research material not found in context."


# ------------------------------------------------------
# Researcher
# ------------------------------------------------------
# Create the agent
# researcher = Agent(
#     role="Researcher",
#     goal="Research the topic based on latest information",
#     backstory="As an expert in the field of {topic}, you will research the topic using the available tools and provide the necessary information",
#     max_iter=3,  # This is the maximum number of iterations that the agent will use to generate the output
#     max_rpm=500,  # This is the maximum number of requests per minute that the agent can make to the language model
#     verbose=True,  # This is a flag that determines if the agent will print more output to the console
#     step_callback=write_result_to_file,  # This is a callback function that will be called after each iteration of the agent
#     allow_delegation=False,  # This is a flag that determines if the agent can delegate the task to another agent. As we are only using one agent, we set this to False
#     tools=ExaSearchToolset.tools(),
#     llm=llm,
# )
researcher = Agent(
    role="Researcher",
    goal="Research topics and provide detailed information",
    backstory="As an expert in the field of {topic}, you will research the topic using the available tools and provide the necessary information",
    verbose=True,
    tools=ExaSearchToolset.tools(),
    memory=False,  # Enable memory
    llm=llm,
)
# Create the task
research_task = Task(
    description="Research the topic",  # This is a description of the task
    agent=researcher,  # This is the agent that will be assigned the task
    expected_output="5 paragraphs of information on the topic",  # This is the expected output of the taskafter its completion
    verbose=True,  # This is a flag that determines if the task will print more output to the console
    post_process=store_in_context,
    output_file="research_result.txt",  # This is the file where the output of the task will be written to, in this case, it is "research_result.txt"
)

# ------------------------------------------------------
# Evaluator
# ------------------------------------------------------
# evaluator = Agent(
#     role="Evaluator",
#     goal="Evaluate the search results provided by the researcher. First, identify the most useful website and then develop 5 key points from the search results and finally generate a 1 paragraph summary of the search results.",
#     backstory="As an expert Research Editor, you are able to read and evaluate research material and to identify the key topics and isses and succinctly. You use well structured formal grammar that is easy to engage with.",
#     max_iter=4,  # This is the maximum number of iterations that the agent will use to generate the output
#     max_rpm=500,  # This is the maximum number of requests per minute that the agent can make to the language model
#     verbose=True,  # This is a flag that determines if the agent will print more output to the console
#     step_callback=write_result_to_file,  # This is a callback function that will be called after each iteration of the agent
#     allow_delegation=False,  # This is a flag that determines if the agent can delegate the task to another agent. As we are only using one agent, we set this to False
#     llm=llm,
# )
evaluator = Agent(
    role="Evaluator",
    goal="Evaluate the research material",
    backstory="As an expert Research Editor, you are able to read and evaluate research material and to identify the key topics and isses and succinctly. You use well structured formal grammar that is easy to engage with.",
    verbose=True,
    memory=False,  # Enable memory
    llm=llm,
)
evaluator_task = Task(
    description="Evaluate the research material and extract the key points in addition to a well structured summary",  # This is a description of the task
    agent=evaluator,  # This is the agent that will be assigned the task
    expected_output="The website title, 5 key points and a 1 paragraph summary of all the relevant research material",  # This is the expected output of the taskafter its completion
    verbose=True,  # This is a flag that determines if the task will print more output to the console
    post_process=check_context_and_summarize,
    context=["research_material"],
    output_file="summary_result.txt",  # This is the file where the output of the task will be written to, in this case, it is "research_result.txt"
)
evaluator_task.context = [research_task]
# Create the crew
crew = Crew(
    agents=[
        researcher,
        evaluator,
    ],  # This is a list of agents that will be part of the crew
    tasks=[
        research_task,
        evaluator_task,
    ],  # This is a list of tasks that the crew will be assigned
    process=Process.sequential,  # This is the process that the crew will use to complete the tasks, in this case, we are using a sequential process
    verbose=True,  # This is a flag that determines if the crew will print more output to the console
    memory=False,  # This is a flag that determines if the crew will use memory to store information about the tasks in a vector database
    cache=False,  # This is a flag that determines if the crew will use a cache. A cache is not needed in this example, so we set this to False
    max_rpm=500,  # This is the maximum number of requests per minute that the crew can make to the language model
)

# Starting start the crew
result = crew.kickoff(
    inputs={"topic": "The history of the GoldBond foot powder."}
)  # Change the topic to whatever you want to research
print("Final Result:", result)
