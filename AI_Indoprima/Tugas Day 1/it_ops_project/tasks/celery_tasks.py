from pathlib import Path
import os
import yaml

from crewai import Agent, Crew, LLM, Process, Task
from tasks.celery_app import app


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"

AGENTS_CONFIG_PATH = CONFIG_DIR / "agents.yaml"
TASKS_CONFIG_PATH = CONFIG_DIR / "tasks.yaml"

CREWAI_DEFAULT_MODEL = os.getenv("CREWAI_DEFAULT_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def load_yaml_config(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not data:
        raise ValueError(f"File konfigurasi kosong: {file_path}")

    return data


def build_llm(model: str | None = None) -> LLM:
    model_name = model or CREWAI_DEFAULT_MODEL

    if model_name.startswith("ollama/"):
        model_to_use = model_name
    else:
        model_to_use = f"ollama/{model_name}"

    print("MODEL_TO_USE:", model_to_use)
    print("OLLAMA_BASE_URL:", OLLAMA_BASE_URL)

    return LLM(
        model=model_to_use,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        timeout=300,
    )


def build_agent(agent_key: str, llm: LLM) -> Agent:
    agents_config = load_yaml_config(AGENTS_CONFIG_PATH)

    if agent_key not in agents_config:
        raise KeyError(f"Agent key tidak ditemukan di agents.yaml: {agent_key}")

    agent_config = agents_config[agent_key]

    return Agent(
        role=agent_config["role"],
        goal=agent_config["goal"],
        backstory=agent_config["backstory"],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_task_description(task_key: str, input_data: dict) -> str:
    tasks_config = load_yaml_config(TASKS_CONFIG_PATH)

    if task_key not in tasks_config:
        raise KeyError(f"Task key tidak ditemukan di tasks.yaml: {task_key}")

    task_config = tasks_config[task_key]

    workflow_text = "\n".join(
        [f"- {item}" for item in task_config.get("workflow", [])]
    )

    output_format_text = "\n".join(
        [f"{index + 1}. {item}" for index, item in enumerate(task_config.get("expected_output_format", []))]
    )

    input_text = "\n".join(
        [f"{key}: {value}" for key, value in input_data.items()]
    )

    return (
        f"Nama task: {task_config.get('name')}\n"
        f"Endpoint terkait: {task_config.get('api_endpoint')}\n\n"
        f"Data input:\n{input_text}\n\n"
        f"Instruksi:\n{task_config.get('instruction')}\n\n"
        f"Workflow analisis:\n{workflow_text}\n\n"
        f"Format output wajib:\n{output_format_text}\n"
    )


def get_expected_output(task_key: str) -> str:
    tasks_config = load_yaml_config(TASKS_CONFIG_PATH)

    if task_key not in tasks_config:
        raise KeyError(f"Task key tidak ditemukan di tasks.yaml: {task_key}")

    task_config = tasks_config[task_key]
    output_format = task_config.get("expected_output_format", [])

    return (
        "Output dalam bahasa Indonesia yang memuat: "
        + ", ".join(output_format)
    )


@app.task(name="tasks.celery_tasks.analyze_incident_task")
def analyze_incident_task(
    requester_name: str,
    system_name: str,
    issue_description: str,
    urgency: str,
    model: str | None = None,
):
    try:
        llm = build_llm(model)

        agent = build_agent(
            agent_key="it_incident_triage_agent",
            llm=llm,
        )

        input_data = {
            "Nama pelapor": requester_name,
            "Sistem terdampak": system_name,
            "Deskripsi masalah": issue_description,
            "Tingkat urgensi": urgency,
        }

        task = Task(
            description=build_task_description(
                task_key="analyze_incident_task",
                input_data=input_data,
            ),
            expected_output=get_expected_output("analyze_incident_task"),
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
            "api_type": "incident_analysis",
            "agent_key": "it_incident_triage_agent",
            "task_key": "analyze_incident_task",
            "requester_name": requester_name,
            "system_name": system_name,
            "urgency": urgency,
            "response": str(result),
        }

    except Exception as e:
        return {
            "status": "error",
            "api_type": "incident_analysis",
            "message": str(e),
            "error_type": type(e).__name__,
        }


@app.task(name="tasks.celery_tasks.review_change_task")
def review_change_task(
    application_name: str,
    change_description: str,
    deployment_time: str,
    rollback_plan: str,
    risk_level: str,
    model: str | None = None,
):
    try:
        llm = build_llm(model)

        agent = build_agent(
            agent_key="it_change_risk_reviewer_agent",
            llm=llm,
        )

        input_data = {
            "Nama aplikasi": application_name,
            "Deskripsi perubahan": change_description,
            "Waktu deployment": deployment_time,
            "Rollback plan": rollback_plan,
            "Level risiko awal": risk_level,
        }

        task = Task(
            description=build_task_description(
                task_key="review_change_task",
                input_data=input_data,
            ),
            expected_output=get_expected_output("review_change_task"),
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
            "api_type": "change_review",
            "agent_key": "it_change_risk_reviewer_agent",
            "task_key": "review_change_task",
            "application_name": application_name,
            "risk_level": risk_level,
            "response": str(result),
        }

    except Exception as e:
        return {
            "status": "error",
            "api_type": "change_review",
            "message": str(e),
            "error_type": type(e).__name__,
        }