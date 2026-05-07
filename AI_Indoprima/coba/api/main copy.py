from fastapi import FastAPI
from pydantic import BaseModel
from celery.result import AsyncResult

from tasks.celery_app import app as celery_app
from tasks.celery_tasks import run_agent_task


app = FastAPI(
    title="CrewAI FastAPI Celery API",
    version="1.0.0"
)


class CrewRequest(BaseModel):
    agent_name: str = "Research Assistant"
    prompt: str
    model: str | None = None
    knowledge_base: str | None = None


@app.get("/")
def root():
    return {
        "message": "CrewAI FastAPI is running"
    }


@app.get("/health")
def health():
    return {
        "fastapi": "ok"
    }


@app.post("/crews/run")
def run_crew(payload: CrewRequest):
    task = run_agent_task.delay(
        payload.agent_name,
        payload.prompt,
        payload.model,
        payload.knowledge_base,
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Task berhasil masuk antrean Celery."
    }


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    if result.status == "PENDING":
        return {
            "task_id": task_id,
            "status": "PENDING",
            "result": None
        }

    if result.status == "STARTED":
        return {
            "task_id": task_id,
            "status": "STARTED",
            "result": None
        }

    if result.status == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "result": result.result
        }

    if result.status == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "error": str(result.result),
            "traceback": result.traceback
        }

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }
