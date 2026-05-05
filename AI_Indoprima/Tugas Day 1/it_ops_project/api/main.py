from fastapi import FastAPI
from pydantic import BaseModel
from celery.result import AsyncResult

from tasks.celery_app import app as celery_app
from tasks.celery_tasks import analyze_incident_task, review_change_task


app = FastAPI(
    title="IT Operations Assistant API",
    version="1.0.0"
)


class IncidentRequest(BaseModel):
    requester_name: str
    system_name: str
    issue_description: str
    urgency: str
    model: str | None = "llama3.2:1b"


class ChangeRequest(BaseModel):
    application_name: str
    change_description: str
    deployment_time: str
    rollback_plan: str
    risk_level: str
    model: str | None = "llama3.2:1b"


@app.get("/")
def root():
    return {
        "message": "IT Operations Assistant API is running",
        "version": "it-ops-v1"
    }


@app.get("/health")
def health():
    return {
        "fastapi": "ok"
    }


@app.post("/it/incidents/analyze")
def analyze_incident(payload: IncidentRequest):
    task = analyze_incident_task.delay(
        payload.requester_name,
        payload.system_name,
        payload.issue_description,
        payload.urgency,
        payload.model,
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Incident analysis task berhasil masuk antrean Celery."
    }


@app.post("/it/changes/review")
def review_change(payload: ChangeRequest):
    task = review_change_task.delay(
        payload.application_name,
        payload.change_description,
        payload.deployment_time,
        payload.rollback_plan,
        payload.risk_level,
        payload.model,
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Change review task berhasil masuk antrean Celery."
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