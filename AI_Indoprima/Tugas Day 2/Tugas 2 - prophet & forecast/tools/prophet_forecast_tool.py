from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from pathlib import Path
from datetime import datetime
from prophet import Prophet

import pandas as pd
import json
import re


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class ProphetForecastInput(BaseModel):
    file_path: str = Field(..., description="Path file time series yang akan diprediksi.")
    original_filename: str = Field(..., description="Nama file asli.")
    date_column: str = Field("auto", description="Nama kolom tanggal. Gunakan auto untuk kolom pertama.")
    value_column: str = Field("auto", description="Nama kolom nilai. Gunakan auto untuk kolom kedua.")
    periods: int = Field(30, description="Jumlah periode ke depan yang akan diprediksi.")
    freq: str = Field("D", description="Frekuensi prediksi. Contoh: D, W, MS, H.")


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def make_json_safe(value):
    if pd.isna(value):
        return None

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


def dataframe_to_safe_records(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    safe_records = []

    for row in records:
        safe_row = {}
        for key, value in row.items():
            safe_row[str(key)] = make_json_safe(value)
        safe_records.append(safe_row)

    return safe_records


class ProphetForecastTool(BaseTool):
    name: str = "Prophet Forecast Tool"
    description: str = (
        "Tool untuk membaca data time series dari Excel atau CSV, "
        "menjalankan prediksi menggunakan Prophet, lalu menyimpan hasil prediksi "
        "ke Excel dan JSON."
    )
    args_schema: Type[BaseModel] = ProphetForecastInput

    def _run(
        self,
        file_path: str,
        original_filename: str,
        date_column: str = "auto",
        value_column: str = "auto",
        periods: int = 30,
        freq: str = "D",
    ) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        if path.suffix.lower() == ".csv":
            raw_df = pd.read_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(path)
        else:
            raise ValueError("Format file harus .csv, .xlsx, atau .xls.")

        if raw_df.shape[1] < 2:
            raise ValueError("File minimal harus memiliki 2 kolom: tanggal dan nilai.")

        raw_df.columns = [str(col).strip() for col in raw_df.columns]

        if date_column == "auto" or date_column not in raw_df.columns:
            date_column = raw_df.columns[0]

        if value_column == "auto" or value_column not in raw_df.columns:
            value_column = raw_df.columns[1]

        df = raw_df[[date_column, value_column]].copy()
        df = df.rename(columns={date_column: "ds", value_column: "y"})

        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        df = df.dropna(subset=["ds", "y"])
        df = df.sort_values("ds")

        df = df.groupby("ds", as_index=False)["y"].mean()

        if len(df) < 3:
            raise ValueError("Data valid terlalu sedikit. Minimal 3 baris data time series.")

        periods = int(periods)
        if periods < 1:
            periods = 30

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        forecast_columns = [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "trend",
        ]

        forecast_result = forecast[forecast_columns].copy()
        future_forecast = forecast_result.tail(periods).copy()

        last_actual_date = df["ds"].max()
        last_actual_value = float(df["y"].iloc[-1])
        forecast_end_date = future_forecast["ds"].max()
        forecast_end_value = float(future_forecast["yhat"].iloc[-1])

        forecast_change = forecast_end_value - last_actual_value
        forecast_change_percent = (
            (forecast_change / last_actual_value) * 100
            if last_actual_value != 0
            else None
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = slugify(Path(original_filename).stem) or "time_series_forecast"

        output_excel = OUTPUT_DIR / f"prophet_forecast_{safe_name}_{timestamp}.xlsx"
        output_json = OUTPUT_DIR / f"prophet_forecast_{safe_name}_{timestamp}.json"

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Original Filename",
                    "Date Column",
                    "Value Column",
                    "Total Historical Data",
                    "Forecast Periods",
                    "Frequency",
                    "Last Actual Date",
                    "Last Actual Value",
                    "Forecast End Date",
                    "Forecast End Value",
                    "Forecast Change",
                    "Forecast Change Percent",
                ],
                "Value": [
                    original_filename,
                    date_column,
                    value_column,
                    len(df),
                    periods,
                    freq,
                    last_actual_date,
                    last_actual_value,
                    forecast_end_date,
                    forecast_end_value,
                    forecast_change,
                    forecast_change_percent,
                ],
            }
        )

        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Historical Data")
            forecast_result.to_excel(writer, index=False, sheet_name="Forecast All")
            future_forecast.to_excel(writer, index=False, sheet_name="Forecast Future")
            summary_df.to_excel(writer, index=False, sheet_name="Summary")

        json_content = {
            "status": "success",
            "tool": "Prophet Forecast Tool",
            "original_filename": original_filename,
            "date_column": date_column,
            "value_column": value_column,
            "total_historical_data": len(df),
            "forecast_periods": periods,
            "frequency": freq,
            "last_actual": {
                "date": make_json_safe(last_actual_date),
                "value": last_actual_value,
            },
            "forecast_end": {
                "date": make_json_safe(forecast_end_date),
                "value": forecast_end_value,
            },
            "forecast_change": forecast_change,
            "forecast_change_percent": forecast_change_percent,
            "forecast_preview": dataframe_to_safe_records(future_forecast.tail(10)),
            "files": {
                "output_excel": str(output_excel),
                "output_json": str(output_json),
            },
            "generated_at": timestamp,
        }

        with output_json.open("w", encoding="utf-8") as file:
            json.dump(json_content, file, ensure_ascii=False, indent=2)

        return json.dumps(json_content, ensure_ascii=False, indent=2)