from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from celery.result import AsyncResult
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from uuid import uuid4
import shutil

from tasks.celery_app import app as celery_app
from tasks.celery_tasks import run_prophet_forecast_task


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Prophet Forecast Agent API",
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
        "message": "Prophet Forecast Agent API is running",
        "version": "prophet-forecast-v1"
    }


@app.get("/health")
def health():
    return {
        "fastapi": "ok"
    }


@app.post("/forecast/run")
async def run_forecast(
    file: UploadFile = File(...),
    date_column: str = Form("auto"),
    value_column: str = Form("auto"),
    periods: int = Form(30),
    freq: str = Form("D"),
    model: str = Form("llama3.2:1b"),
):
    safe_filename = file.filename.replace("/", "_").replace("\\", "_")
    unique_filename = f"{uuid4()}_{safe_filename}"
    file_path = UPLOAD_DIR / unique_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = run_prophet_forecast_task.delay(
        str(file_path),
        file.filename,
        date_column,
        value_column,
        periods,
        freq,
        model,
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Prophet forecast task berhasil masuk antrean Celery.",
        "uploaded_file": file.filename,
        "date_column": date_column,
        "value_column": value_column,
        "periods": periods,
        "freq": freq,
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


@app.get("/download")
def download_file(path: str):
    file_path = Path(path)

    if not file_path.exists():
        return {
            "status": "error",
            "message": "File tidak ditemukan."
        }

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )