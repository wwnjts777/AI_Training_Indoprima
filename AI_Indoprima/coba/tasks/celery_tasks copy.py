from tasks.celery_app import app

@app.task(name="tasks.celery_tasks.run_agent_task")
def run_agent_task(
    agent_name: str = "Research Assistant",
    prompt: str = "",
    model: str | None = None,
    knowledge_base: str | None = None,
):
    return {
        "status": "success",
        "agent_name": agent_name,
        "prompt": prompt,
        "model": model,
        "knowledge_base": knowledge_base,
        "message": "Celery task berhasil dikenali dan dijalankan."
    }
