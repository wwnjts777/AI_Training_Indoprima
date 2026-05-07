from tasks.celery_app import app
#from src.indoprimaflow.crews.crew_latihan.crew_latihan import CrewLatihan
from src.coba.crews.content_crew.content_crew import ContentCrew
#from src.indoprimaflow.crews.crew_latihan.crew_latihan import Crew
from crewai import Crew, Agent, Task, Process, LLM
#from src.indoprimaflow.crews.content_crew.content_crew import ContentCrew
from src.coba.crews.content_crew.content_crew import ContentCrew
import logging
import traceback

logger = logging.getLogger(__name__)


#@celery_app.task(bind=True, name="research")
#def research(self, topic:str):
#    self.update_state(state='RUNNING', meta={'current':f'start job for{topic}'})
#    try:
#        result = ContentCrew().crew().kickoff_async(inputs = {"topic": topic})
#        return str(result)
#    except Exception as e:
#        # self.update_state(state='FAILURE', meta={'error':str(e)})
#        logger.error(f"Task failed with error: {e}\n{traceback.format_exc()}")
#        raise

from tasks.celery_app import app
from crewai import Agent, Task, Crew, Process, LLM
import os


CREWAI_DEFAULT_MODEL = os.getenv("CREWAI_DEFAULT_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@app.task(name="tasks.celery_tasks.run_agent_task")
def run_agent_task(
    agent_name: str = "Research Assistant",
    prompt: str = "",
    model: str | None = None,
    knowledge_base: str | None = None,
):
    try:
        if not prompt:
            return {
                "status": "error",
                "message": "Prompt tidak boleh kosong."
            }

        model_name = model or CREWAI_DEFAULT_MODEL

        llm = LLM(
            model=f"ollama/{model_name}",
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            timeout=300,
        )

        agent = Agent(
            role=agent_name,
            goal="Menjawab permintaan pengguna secara jelas, ringkas, dan sistematis.",
            backstory=(
                "Anda adalah asisten AI lokal berbasis Ollama yang membantu "
                "menjelaskan topik teknis dengan bahasa sederhana."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        task_description = prompt

        if knowledge_base:
            task_description += f"\n\nKonteks tambahan:\n{knowledge_base}"

        task = Task(
            description=task_description,
            expected_output="Jawaban akhir yang jelas, ringkas, dan mudah dipahami.",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        return {
            "status": "success",
            "agent_name": agent_name,
            "model": f"ollama/{model_name}",
            "response": str(result),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }