from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "prophet_forecast_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

app.conf.update(
    imports=("tasks.celery_tasks",),
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
)