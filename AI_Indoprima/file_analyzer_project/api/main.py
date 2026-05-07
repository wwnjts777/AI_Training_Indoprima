from fastapi import FastAPI, UploadFile, File, Form
from celery.result import AsyncResult
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from uuid import uuid4
import shutil

from tasks.celery_app import app as celery_app
from tasks.celery_tasks import analyze_file_task


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="File Analyzer CrewAI API",
    version="1.0.0"
)


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[dict | str] = None
    error: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "File Analyzer CrewAI API is running",
        "version": "file-analyzer-v1"
    }


@app.get("/health")
def health():
    return {
        "fastapi": "ok"
    }


@app.post("/files/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    analysis_type: str = Form("summary"),
    model: str = Form("llama3.2:1b"),
):
    safe_filename = file.filename.replace("/", "_").replace("\\", "_")
    unique_filename = f"{uuid4()}_{safe_filename}"
    file_path = UPLOAD_DIR / unique_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = analyze_file_task.delay(
        str(file_path),
        file.filename,
        analysis_type,
        model,
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "File analysis task berhasil masuk antrean Celery.",
        "uploaded_file": file.filename,
        "saved_path": str(file_path),
    }


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.state,
        "result": None,
        "error": None,
    }

    if task_result.state == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.state == "FAILURE":
        response["error"] = str(task_result.info)

    return response