from tasks.celery_app import app
from crewai import Agent, Crew, Task, Process, LLM
from tools.prophet_forecast_tool import ProphetForecastTool

from pathlib import Path
from datetime import datetime
import os
import json
import traceback
import logging


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CREWAI_DEFAULT_MODEL = os.getenv("CREWAI_DEFAULT_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def build_llm(model: str | None = None) -> LLM:
    model_name = model or CREWAI_DEFAULT_MODEL

    if model_name.startswith("ollama/"):
        model_to_use = model_name
    else:
        model_to_use = f"ollama/{model_name}"

    return LLM(
        model=model_to_use,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        timeout=300,
    )


def save_markdown_report(
    original_filename: str,
    forecast_result: dict,
    agent_report: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(original_filename).stem.replace(" ", "_").lower()

    output_md = OUTPUT_DIR / f"prophet_forecast_report_{safe_name}_{timestamp}.md"

    markdown_content = f"""# Report Prediksi Time Series Prophet

## Nama File

{original_filename}

---

## Ringkasan Hasil Forecast

- Total data historis: {forecast_result.get("total_historical_data")}
- Periode prediksi: {forecast_result.get("forecast_periods")}
- Frekuensi: {forecast_result.get("frequency")}
- Nilai aktual terakhir: {forecast_result.get("last_actual", {}).get("value")}
- Nilai prediksi akhir: {forecast_result.get("forecast_end", {}).get("value")}
- Perubahan prediksi: {forecast_result.get("forecast_change")}
- Persentase perubahan: {forecast_result.get("forecast_change_percent")}

---

## Report Agent

{agent_report}

---

Generated at: {timestamp}
"""

    output_md.write_text(markdown_content, encoding="utf-8")

    return str(output_md)


@app.task(bind=True, name="tasks.celery_tasks.run_prophet_forecast_task")
def run_prophet_forecast_task(
    self,
    file_path: str,
    original_filename: str,
    date_column: str = "auto",
    value_column: str = "auto",
    periods: int = 30,
    freq: str = "D",
    model: str | None = None,
):
    try:
        self.update_state(
            state="RUNNING",
            meta={"current": f"Mulai prediksi time series untuk {original_filename}"}
        )

        prophet_tool = ProphetForecastTool()

        tool_result_text = prophet_tool._run(
            file_path=file_path,
            original_filename=original_filename,
            date_column=date_column,
            value_column=value_column,
            periods=periods,
            freq=freq,
        )

        forecast_result = json.loads(tool_result_text)

        llm = build_llm(model)

        report_agent = Agent(
            role="Time Series Forecast Report Analyst",
            goal=(
                "Membuat report prediksi time series berdasarkan hasil forecast "
                "dari custom tool Prophet."
            ),
            backstory=(
                "Anda adalah analis data yang memahami forecasting time series, "
                "tren historis, hasil prediksi Prophet, interpretasi yhat, "
                "rentang ketidakpastian, dan rekomendasi bisnis atau operasional."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        report_task = Task(
            description=(
                "Buat report prediksi dalam bahasa Indonesia berdasarkan hasil custom tool berikut:\n\n"
                f"{json.dumps(forecast_result, ensure_ascii=False, indent=2)}\n\n"
                "Gunakan format:\n"
                "1. Ringkasan umum\n"
                "2. Interpretasi tren prediksi\n"
                "3. Nilai aktual terakhir dan prediksi akhir\n"
                "4. Potensi kenaikan atau penurunan\n"
                "5. Catatan risiko prediksi\n"
                "6. Rekomendasi tindak lanjut\n"
                "7. Kesimpulan"
            ),
            expected_output=(
                "Report prediksi time series dalam bahasa Indonesia yang jelas, ringkas, "
                "dan mudah dipahami."
            ),
            agent=report_agent,
        )

        crew = Crew(
            agents=[report_agent],
            tasks=[report_task],
            process=Process.sequential,
            verbose=True,
        )

        crew_result = crew.kickoff()

        markdown_file = save_markdown_report(
            original_filename=original_filename,
            forecast_result=forecast_result,
            agent_report=str(crew_result),
        )

        return {
            "status": "success",
            "api_type": "prophet_time_series_forecast",
            "original_filename": original_filename,
            "date_column": forecast_result.get("date_column"),
            "value_column": forecast_result.get("value_column"),
            "forecast_periods": forecast_result.get("forecast_periods"),
            "frequency": forecast_result.get("frequency"),
            "last_actual": forecast_result.get("last_actual"),
            "forecast_end": forecast_result.get("forecast_end"),
            "forecast_change": forecast_result.get("forecast_change"),
            "forecast_change_percent": forecast_result.get("forecast_change_percent"),
            "forecast_preview": forecast_result.get("forecast_preview"),
            "output_excel": forecast_result.get("files", {}).get("output_excel"),
            "output_json": forecast_result.get("files", {}).get("output_json"),
            "markdown_file": markdown_file,
            "agent_report": str(crew_result),
        }

    except Exception as e:
        logger.error(f"Task failed with error: {e}\n{traceback.format_exc()}")

        return {
            "status": "error",
            "api_type": "prophet_time_series_forecast",
            "original_filename": original_filename,
            "message": str(e),
            "error_type": type(e).__name__,
        }