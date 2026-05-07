from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task


ollama_llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.2,
    timeout=300,
)