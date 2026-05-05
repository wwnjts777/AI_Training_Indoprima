from tasks.celery_app import app
from crewai import Agent, Crew, Task, Process, LLM

from pathlib import Path
from datetime import datetime
import os
import re
import json
import traceback
import logging
import pandas as pd
import yaml

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CONFIG_DIR = BASE_DIR / "config"

AGENT_CONFIG_PATH = CONFIG_DIR / "agent.yaml"
TASK_CONFIG_PATH = CONFIG_DIR / "tasks.yaml"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CREWAI_DEFAULT_MODEL = os.getenv("CREWAI_DEFAULT_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_CONTAMINATION = 0.05
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100

ANOMALY_DISPLAY_RATIO = 0.5
MAX_ANOMALY_PREVIEW_ROWS = 50


def build_llm(model: str | None = None) -> LLM:
    model_name = model or CREWAI_DEFAULT_MODEL
    model_to_use = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"

    return LLM(
        model=model_to_use,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        timeout=300,
    )


def load_yaml_config(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not data:
        raise ValueError(f"File konfigurasi kosong: {file_path}")

    return data


def build_agent_from_yaml(agent_key: str, llm: LLM) -> Agent:
    agent_config = load_yaml_config(AGENT_CONFIG_PATH)

    if agent_key not in agent_config:
        raise KeyError(f"Agent key tidak ditemukan di agent.yaml: {agent_key}")

    selected_agent = agent_config[agent_key]

    return Agent(
        role=selected_agent["role"],
        goal=selected_agent["goal"],
        backstory=selected_agent["backstory"],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_task_prompt_from_yaml(
    task_key: str,
    original_filename: str,
    processed_result: dict,
) -> str:
    task_config = load_yaml_config(TASK_CONFIG_PATH)

    if task_key not in task_config:
        raise KeyError(f"Task key tidak ditemukan di tasks.yaml: {task_key}")

    selected_task = task_config[task_key]
    output_sections = selected_task.get("output_sections", [])
    output_section_text = "\n".join(
        [f"{index + 1}. {section}" for index, section in enumerate(output_sections)]
    )

    return (
        f"Nama task: {selected_task.get('name')}\n"
        f"Deskripsi task: {selected_task.get('description')}\n\n"
        f"Nama file: {original_filename}\n"
        f"Metode deteksi anomali: Isolation Forest\n"
        f"Contamination: {processed_result['contamination']}\n"
        f"Random state: {processed_result['random_state']}\n"
        f"Number of estimators: {processed_result['n_estimators']}\n"
        f"Kolom waktu: {processed_result['time_column']}\n"
        f"Kolom sensor: {processed_result['sensor_columns']}\n"
        f"Total data: {processed_result['total_data']}\n"
        f"Batas setengah total data: {processed_result['half_total_data']}\n"
        f"Total baris normal: {processed_result['total_normal_rows']}\n"
        f"Total baris anomaly: {processed_result['total_anomaly_rows']}\n"
        f"Apakah anomaly lebih besar dari total data / 2: {processed_result['show_anomaly_data']}\n"
        f"Keterangan anomaly: {processed_result['anomaly_message']}\n"
        f"Data anomaly yang ditampilkan: {processed_result['anomaly_rows']}\n"
        f"Ringkasan parameter: {processed_result['sensor_summary']}\n"
        f"Preview data: {processed_result['preview_rows']}\n\n"
        "Buat ringkasan analisis dalam bahasa Indonesia hanya dengan format berikut:\n"
        f"{output_section_text}\n"
    )


def get_expected_output_from_yaml(task_key: str) -> str:
    task_config = load_yaml_config(TASK_CONFIG_PATH)

    if task_key not in task_config:
        raise KeyError(f"Task key tidak ditemukan di tasks.yaml: {task_key}")

    return task_config[task_key].get(
        "expected_output",
        "Ringkasan analisis anomali proses dalam bahasa Indonesia.",
    )


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    columns_to_drop = []

    for column in cleaned_df.columns:
        column_name = str(column).strip().lower()
        if column_name.startswith("unnamed"):
            columns_to_drop.append(column)

    if columns_to_drop:
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)

    cleaned_df.columns = [str(col).strip() for col in cleaned_df.columns]
    return cleaned_df


def detect_time_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        column_name = str(column).strip().lower()
        if column_name in ["time", "tanggal", "date", "datetime", "timestamp"]:
            return column

    return df.columns[0] if len(df.columns) > 0 else None


def get_sensor_columns(df: pd.DataFrame, time_column: str | None) -> list[str]:
    sensor_columns = []

    for column in df.columns:
        if column == time_column:
            continue

        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().sum() > 0:
            sensor_columns.append(column)

    if not sensor_columns:
        raise ValueError("Tidak ditemukan kolom sensor numerik untuk dianalisis.")

    return sensor_columns


def make_json_safe(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)

    return value


def make_json_serializable(value):
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    return make_json_safe(value)


def dataframe_preview(df: pd.DataFrame, rows: int = 10) -> list[dict]:
    preview = df.head(rows).to_dict(orient="records")
    safe_preview = []

    for row in preview:
        safe_row = {}
        for key, value in row.items():
            safe_row[str(key)] = make_json_safe(value)
        safe_preview.append(safe_row)

    return safe_preview


def get_anomaly_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Status Overall" not in df.columns:
        return pd.DataFrame()

    return df[df["Status Overall"] == "Abnormality"].copy()


def should_show_anomaly_data(total_anomaly_rows: int, total_data: int) -> bool:
    if total_data == 0:
        return False

    return total_anomaly_rows > (total_data * ANOMALY_DISPLAY_RATIO)


def build_anomaly_reason(row, sensor_columns: list[str], normal_reference: dict) -> str:
    if row["Status Overall"] == "Normal":
        return "Normal"

    reasons = []

    for column in sensor_columns:
        value = row[column]
        mean_value = normal_reference[column]["mean"]
        std_value = normal_reference[column]["std"]

        if pd.isna(value) or pd.isna(mean_value) or pd.isna(std_value):
            continue

        upper_reference = mean_value + (2 * std_value)
        lower_reference = mean_value - (2 * std_value)

        if value > upper_reference:
            reasons.append(f"{column} cenderung tinggi dibanding pola normal")
        elif value < lower_reference:
            reasons.append(f"{column} cenderung rendah dibanding pola normal")

    if not reasons:
        return "Kombinasi parameter tidak sesuai dengan pola operasi normal"

    return "; ".join(reasons)


def build_maintenance_recommendation(row) -> str:
    reason = str(row["Alasan Anomali"]).lower()

    if row["Status Overall"] == "Normal":
        return "Lanjutkan monitoring proses secara rutin. Tidak diperlukan tindakan korektif segera."

    recommendations = []

    if "temp" in reason:
        recommendations.append(
            "Periksa kestabilan temperatur proses, sensor temperatur, potensi overheating, kondisi material, dan abnormalitas operasi cyclone."
        )

    if "draft" in reason:
        recommendations.append(
            "Periksa sistem draft, potensi kebocoran udara, plugging, fan, damper, ducting, dan pressure transmitter."
        )

    if not recommendations:
        recommendations.append(
            "Lakukan inspeksi lapangan pada parameter proses yang tidak sesuai pola normal dan validasi ulang pembacaan sensor."
        )

    return " ".join(recommendations)


def build_follow_up_action(row) -> str:
    if row["Status Overall"] == "Normal":
        return "Simpan data sebagai histori monitoring dan lanjutkan inspeksi sesuai jadwal."

    return (
        "Buat work order inspeksi, lakukan pengukuran ulang, validasi sensor, "
        "dan pantau tren parameter pada periode berikutnya."
    )


def process_anomaly_file(
    file_path: str,
    original_filename: str,
    contamination: float,
    random_state: int,
    n_estimators: int,
) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    if path.suffix.lower() not in [".xlsx", ".xls"]:
        raise ValueError("Format file harus .xlsx atau .xls")

    df = pd.read_excel(path)
    df = clean_dataframe(df)

    if df.empty:
        raise ValueError("File kosong atau tidak memiliki data.")

    time_column = detect_time_column(df)
    sensor_columns = get_sensor_columns(df, time_column)

    model_df = df[sensor_columns].copy()

    for column in sensor_columns:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.fillna(model_df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(model_df)

    isolation_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    anomaly_label = isolation_model.fit_predict(scaled_data)
    anomaly_score = isolation_model.decision_function(scaled_data)

    df["Anomaly Label"] = anomaly_label
    df["Anomaly Score"] = anomaly_score
    df["Status Overall"] = df["Anomaly Label"].apply(
        lambda label: "Abnormality" if label == -1 else "Normal"
    )

    normal_rows = df[df["Status Overall"] == "Normal"]

    normal_reference = {}
    for column in sensor_columns:
        numeric_series = pd.to_numeric(normal_rows[column], errors="coerce")
        normal_reference[column] = {
            "mean": numeric_series.mean(),
            "std": numeric_series.std(),
            "min": numeric_series.min(),
            "max": numeric_series.max(),
        }

    df["Alasan Anomali"] = df.apply(
        lambda row: build_anomaly_reason(row, sensor_columns, normal_reference),
        axis=1,
    )
    df["Saran Maintenance"] = df.apply(build_maintenance_recommendation, axis=1)
    df["Tindakan Lanjutan"] = df.apply(build_follow_up_action, axis=1)

    total_data = len(df)
    total_anomaly_rows = int((df["Status Overall"] == "Abnormality").sum())
    total_normal_rows = int((df["Status Overall"] == "Normal").sum())

    anomaly_df = get_anomaly_dataframe(df)
    show_anomaly_data = should_show_anomaly_data(total_anomaly_rows, total_data)

    if show_anomaly_data:
        anomaly_message = (
            f"Jumlah data anomaly {total_anomaly_rows} lebih besar dari setengah total data "
            f"({total_data} / 2 = {total_data / 2}). Data anomaly ditampilkan."
        )
    else:
        anomaly_message = (
            f"Jumlah data anomaly {total_anomaly_rows} tidak lebih besar dari setengah total data "
            f"({total_data} / 2 = {total_data / 2}). Data anomaly tidak ditampilkan penuh."
        )

    sensor_summary_rows = []
    for column in sensor_columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        sensor_summary_rows.append(
            {
                "Parameter": column,
                "Mean": numeric_series.mean(),
                "Min": numeric_series.min(),
                "Max": numeric_series.max(),
                "Std": numeric_series.std(),
                "Normal Reference Mean": normal_reference[column]["mean"],
                "Normal Reference Std": normal_reference[column]["std"],
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = slugify(Path(original_filename).stem) or "process_anomaly"
    output_excel = OUTPUT_DIR / f"hasil_isolation_forest_{safe_name}_{timestamp}.xlsx"

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil Analisis")

        if show_anomaly_data:
            anomaly_df.to_excel(writer, index=False, sheet_name="Data Anomaly")

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Data",
                    "Batas Setengah Total Data",
                    "Total Baris Normal",
                    "Total Baris Abnormality",
                    "Anomaly Lebih Besar dari Total Data / 2",
                    "Metode Deteksi",
                    "Contamination",
                    "Random State",
                    "Number of Estimators",
                    "Keterangan Anomaly",
                ],
                "Value": [
                    total_data,
                    total_data / 2,
                    total_normal_rows,
                    total_anomaly_rows,
                    "Ya" if show_anomaly_data else "Tidak",
                    "Isolation Forest",
                    contamination,
                    random_state,
                    n_estimators,
                    anomaly_message,
                ],
            }
        )
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

        sensor_summary_df = pd.DataFrame(sensor_summary_rows)
        sensor_summary_df.to_excel(writer, index=False, sheet_name="Sensor Summary")

    return {
        "output_excel": str(output_excel),
        "time_column": time_column,
        "sensor_columns": sensor_columns,
        "method": "isolation_forest",
        "contamination": contamination,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "total_data": total_data,
        "half_total_data": total_data / 2,
        "total_anomaly_rows": total_anomaly_rows,
        "total_normal_rows": total_normal_rows,
        "show_anomaly_data": show_anomaly_data,
        "anomaly_message": anomaly_message,
        "anomaly_rows": dataframe_preview(anomaly_df, MAX_ANOMALY_PREVIEW_ROWS) if show_anomaly_data else [],
        "sensor_summary": sensor_summary_rows,
        "preview_rows": dataframe_preview(df),
    }


def save_markdown_summary(original_filename: str, analysis_result: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = slugify(Path(original_filename).stem) or "process_anomaly"
    output_md = OUTPUT_DIR / f"ringkasan_isolation_forest_{safe_name}_{timestamp}.md"

    markdown_content = f"""# Ringkasan Analisis Anomali Proses

## Nama File

{original_filename}

---

## Metode

Isolation Forest

---

## Hasil Analisis

{analysis_result}

---

Generated at: {timestamp}
"""

    output_md.write_text(markdown_content, encoding="utf-8")
    return str(output_md)


def save_json_output(
    original_filename: str,
    processed_result: dict,
    markdown_file: str,
    crew_summary: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = slugify(Path(original_filename).stem) or "process_anomaly"
    output_json = OUTPUT_DIR / f"hasil_isolation_forest_{safe_name}_{timestamp}.json"

    json_content = {
        "status": "success",
        "api_type": "process_anomaly_analysis",
        "original_filename": original_filename,
        "method": processed_result["method"],
        "parameters": {
            "contamination": processed_result["contamination"],
            "random_state": processed_result["random_state"],
            "n_estimators": processed_result["n_estimators"],
        },
        "columns": {
            "time_column": processed_result["time_column"],
            "sensor_columns": processed_result["sensor_columns"],
        },
        "summary": {
            "total_data": processed_result["total_data"],
            "half_total_data": processed_result["half_total_data"],
            "total_normal_rows": processed_result["total_normal_rows"],
            "total_anomaly_rows": processed_result["total_anomaly_rows"],
            "show_anomaly_data": processed_result["show_anomaly_data"],
            "anomaly_message": processed_result["anomaly_message"],
        },
        "anomaly_rows": processed_result["anomaly_rows"],
        "sensor_summary": processed_result["sensor_summary"],
        "preview_rows": processed_result["preview_rows"],
        "files": {
            "output_excel": processed_result["output_excel"],
            "markdown_file": markdown_file,
            "json_file": str(output_json),
        },
        "crew_summary": crew_summary,
        "generated_at": timestamp,
    }

    json_content = make_json_serializable(json_content)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(json_content, file, ensure_ascii=False, indent=2)

    return str(output_json)


@app.task(bind=True, name="tasks.celery_tasks.analyze_process_anomaly_task")
def analyze_process_anomaly_task(
    self,
    file_path: str,
    original_filename: str,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    model: str | None = None,
):
    try:
        self.update_state(
            state="RUNNING",
            meta={"current": f"Mulai menganalisis file {original_filename}"},
        )

        try:
            contamination = float(contamination)
        except (ValueError, TypeError):
            contamination = DEFAULT_CONTAMINATION

        if contamination <= 0 or contamination >= 0.5:
            contamination = DEFAULT_CONTAMINATION

        try:
            random_state = int(random_state)
        except (ValueError, TypeError):
            random_state = DEFAULT_RANDOM_STATE

        try:
            n_estimators = int(n_estimators)
        except (ValueError, TypeError):
            n_estimators = DEFAULT_N_ESTIMATORS

        if n_estimators < 10:
            n_estimators = DEFAULT_N_ESTIMATORS

        processed_result = process_anomaly_file(
            file_path=file_path,
            original_filename=original_filename,
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
        )

        llm = build_llm(model)

        agent = build_agent_from_yaml(
            agent_key="process_anomaly_analyst",
            llm=llm,
        )

        summary_prompt = build_task_prompt_from_yaml(
            task_key="process_anomaly_summary_task",
            original_filename=original_filename,
            processed_result=processed_result,
        )

        task = Task(
            description=summary_prompt,
            expected_output=get_expected_output_from_yaml("process_anomaly_summary_task"),
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        crew_result = crew.kickoff()

        markdown_file = save_markdown_summary(
            original_filename=original_filename,
            analysis_result=str(crew_result),
        )

        json_file = save_json_output(
            original_filename=original_filename,
            processed_result=processed_result,
            markdown_file=markdown_file,
            crew_summary=str(crew_result),
        )

        return {
            "status": "success",
            "api_type": "process_anomaly_analysis",
            "original_filename": original_filename,
            "method": processed_result["method"],
            "contamination": processed_result["contamination"],
            "random_state": processed_result["random_state"],
            "n_estimators": processed_result["n_estimators"],
            "time_column": processed_result["time_column"],
            "sensor_columns": processed_result["sensor_columns"],
            "total_data": processed_result["total_data"],
            "half_total_data": processed_result["half_total_data"],
            "total_normal_rows": processed_result["total_normal_rows"],
            "total_anomaly_rows": processed_result["total_anomaly_rows"],
            "show_anomaly_data": processed_result["show_anomaly_data"],
            "anomaly_message": processed_result["anomaly_message"],
            "anomaly_rows": processed_result["anomaly_rows"],
            "output_excel": processed_result["output_excel"],
            "markdown_file": markdown_file,
            "json_file": json_file,
            "summary": str(crew_result),
        }

    except Exception as e:
        logger.error(f"Task failed with error: {e}\n{traceback.format_exc()}")

        return {
            "status": "error",
            "api_type": "process_anomaly_analysis",
            "original_filename": original_filename,
            "message": str(e),
            "error_type": type(e).__name__,
        }
