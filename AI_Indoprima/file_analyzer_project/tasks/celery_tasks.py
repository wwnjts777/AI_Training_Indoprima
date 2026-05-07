from tasks.celery_app import app
from crewai import Agent, Crew, Task, Process, LLM

from pathlib import Path
from datetime import datetime
import os
import re
import json
import traceback
import logging

from pypdf import PdfReader
from docx import Document


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CREWAI_DEFAULT_MODEL = os.getenv("CREWAI_DEFAULT_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


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


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_pdf_file(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            text_parts.append(f"\n\n--- Halaman {page_number} ---\n{text}")

    if not text_parts:
        raise ValueError("PDF tidak memiliki teks yang dapat diekstrak.")

    return "\n".join(text_parts)


def read_docx_file(file_path: Path) -> str:
    document = Document(str(file_path))
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    if not paragraphs:
        raise ValueError("File DOCX kosong atau tidak memiliki teks yang dapat dibaca.")

    return "\n".join(paragraphs)


def read_json_file(file_path: Path) -> str:
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    try:
        data = json.loads(content)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content


def read_document_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    allowed_extensions = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".py",
        ".log",
        ".yaml",
        ".yml",
        ".pdf",
        ".docx",
    }

    suffix = path.suffix.lower()

    if suffix not in allowed_extensions:
        raise ValueError(
            f"Format file {suffix} belum didukung. "
            "Gunakan .txt, .md, .csv, .json, .py, .log, .yaml, .yml, .pdf, atau .docx."
        )

    if suffix == ".pdf":
        content = read_pdf_file(path)
    elif suffix == ".docx":
        content = read_docx_file(path)
    elif suffix == ".json":
        content = read_json_file(path)
    else:
        content = path.read_text(encoding="utf-8", errors="ignore")

    if not content.strip():
        raise ValueError("File kosong atau tidak memiliki teks yang dapat dibaca.")

    max_chars = 12000

    if len(content) > max_chars:
        content = (
            content[:max_chars]
            + "\n\n[Catatan: isi dokumen dipotong karena terlalu panjang agar tetap ringan diproses.]"
        )

    return content


def save_analysis_to_markdown(
    original_filename: str,
    analysis_type: str,
    analysis_result: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = slugify(Path(original_filename).stem)[:60] or "document_analysis"

    output_file = OUTPUT_DIR / f"expert_analysis_{safe_name}_{timestamp}.md"

    markdown_content = f"""# Expert Document Analysis

## Nama Dokumen

{original_filename}

## Jenis Analisis

{analysis_type}

---

## Hasil Analisis

{analysis_result}

---

Generated at: {timestamp}
"""

    output_file.write_text(markdown_content, encoding="utf-8")

    return str(output_file)


@app.task(bind=True, name="tasks.celery_tasks.analyze_file_task")
def analyze_file_task(
    self,
    file_path: str,
    original_filename: str,
    analysis_type: str,
    model: str | None = None,
):
    try:
        self.update_state(
            state="RUNNING",
            meta={"current": f"Mulai menganalisis dokumen {original_filename}"}
        )

        document_content = read_document_file(file_path)
        llm = build_llm(model)

        document_analysis_agent = Agent(
            role="Expert Document Analysis Agent",
            goal=(
                "Menganalisis isi dokumen secara mendalam, kritis, sistematis, "
                "dan menghasilkan insight yang dapat digunakan untuk pengambilan keputusan."
            ),
            backstory=(
                "Anda adalah analis dokumen profesional. "
                "Anda berpengalaman membaca proposal, laporan, SOP, dokumen teknis, "
                "dokumen akademik, dokumen bisnis, dokumen kebijakan, dan data operasional. "
                "Anda mampu menemukan tujuan dokumen, struktur isi, isu utama, risiko, gap, "
                "rekomendasi, serta tindakan lanjutan yang relevan."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        expert_document_task = Task(
            description=(
                f"Nama dokumen: {original_filename}\n"
                f"Jenis analisis yang diminta: {analysis_type}\n\n"
                "Berikut isi dokumen yang harus dianalisis:\n"
                "==================================================\n"
                f"{document_content}\n"
                "==================================================\n\n"
                "Lakukan analisis isi dokumen secara expert dalam bahasa Indonesia.\n\n"
                "Gunakan format output berikut:\n\n"
                "1. Identitas Dokumen\n"
                "   - Nama file\n"
                "   - Perkiraan jenis dokumen\n"
                "   - Topik utama\n\n"
                "2. Executive Summary\n"
                "   - Ringkas isi utama dokumen dalam 1 sampai 2 paragraf\n"
                "   - Jelaskan inti informasi yang paling penting\n\n"
                "3. Tujuan Dokumen\n"
                "   - Jelaskan tujuan utama dokumen\n"
                "   - Jelaskan sasaran pembaca atau pengguna dokumen\n\n"
                "4. Struktur dan Isi Utama\n"
                "   - Uraikan bagian-bagian penting dokumen\n"
                "   - Jelaskan alur pembahasan dokumen\n\n"
                "5. Poin Penting\n"
                "   - Buat poin-poin utama secara ringkas\n"
                "   - Hindari pengulangan\n\n"
                "6. Temuan Utama\n"
                "   - Jelaskan temuan penting, pola, data, atau informasi utama\n"
                "   - Bedakan fakta yang disebutkan dokumen dan interpretasi analis\n\n"
                "7. Masalah, Risiko, atau Gap\n"
                "   - Identifikasi kekurangan isi, inkonsistensi, risiko, atau bagian yang belum jelas\n"
                "   - Jika tidak ada, jelaskan bahwa dokumen relatif konsisten\n\n"
                "8. Analisis Kritis\n"
                "   - Nilai kekuatan dokumen\n"
                "   - Nilai kelemahan dokumen\n"
                "   - Jelaskan apakah dokumen sudah cukup kuat untuk digunakan\n\n"
                "9. Rekomendasi Perbaikan\n"
                "   - Berikan rekomendasi praktis dan spesifik\n"
                "   - Susun berdasarkan prioritas\n\n"
                "10. Action Items\n"
                "   - Buat daftar tindakan lanjutan\n"
                "   - Gunakan format: tindakan, PIC yang disarankan, prioritas\n\n"
                "11. Kesimpulan Akhir\n"
                "   - Berikan kesimpulan singkat, jelas, dan tegas"
            ),
            expected_output=(
                "Analisis isi dokumen yang mendalam, kritis, sistematis, dan berbahasa Indonesia."
            ),
            agent=document_analysis_agent,
        )

        crew = Crew(
            agents=[document_analysis_agent],
            tasks=[expert_document_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        markdown_file = save_analysis_to_markdown(
            original_filename=original_filename,
            analysis_type=analysis_type,
            analysis_result=str(result),
        )

        return {
            "status": "success",
            "api_type": "expert_document_analysis",
            "agent": "Expert Document Analysis Agent",
            "original_filename": original_filename,
            "analysis_type": analysis_type,
            "markdown_file": markdown_file,
            "response": str(result),
        }

    except Exception as e:
        logger.error(f"Task failed with error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "api_type": "expert_document_analysis",
            "original_filename": original_filename,
            "message": str(e),
            "error_type": type(e).__name__,
        }