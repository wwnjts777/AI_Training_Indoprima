from tasks.celery_app import app
from crewai import Crew, Agent, Task, Process, LLM
import os


CREWAI_DEFAULT_MODEL = os.getenv("CREWAI_DEFAULT_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@app.task(name="tasks.celery_tasks.run_agent_task")
def run_agent_task(
    agent_name: str = "Research Assistant",
    project_name: str = "Project AI",
    topics: list[str] | None = None,
    model: str | None = None,
    knowledge_base: str | None = None,
):
    try:
        if not topics or len(topics) != 2:
            return {
                "status": "error",
                "message": "Input topics wajib berisi tepat 2 topic."
            }

        if not all(isinstance(topic, str) and topic.strip() for topic in topics):
            return {
                "status": "error",
                "message": "Setiap topic wajib berupa teks dan tidak boleh kosong."
            }

        topic_1 = topics[0].strip()
        topic_2 = topics[1].strip()

        model_name = model or CREWAI_DEFAULT_MODEL

        if model_name.startswith("ollama/"):
            model_to_use = model_name
        else:
            model_to_use = f"ollama/{model_name}"

        print("CELERY TASK FILE ACTIVE")
        print("MODEL_TO_USE:", model_to_use)
        print("OLLAMA_BASE_URL:", OLLAMA_BASE_URL)

        llm = LLM(
            model=model_to_use,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            timeout=300,
        )

        research_agent = Agent(
            role=agent_name,
            goal=(
                "Menjelaskan setiap topic dalam bahasa Indonesia secara jelas, "
                "ringkas, dan sistematis."
            ),
            backstory=(
                "Anda adalah asisten AI lokal berbasis Ollama. "
                "Anda bertugas menjelaskan topic bebas dari pengguna dengan bahasa Indonesia "
                "yang sederhana, runtut, dan mudah dipahami."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        review_agent = Agent(
            role="Review Point Maker",
            goal="Membuat poin-poin review dari jawaban utama untuk setiap topic.",
            backstory=(
                "Anda adalah reviewer yang membaca jawaban utama, lalu menyusun ringkasan, "
                "poin penting, kelebihan, catatan perbaikan, dan kesimpulan singkat."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        resume_agent = Agent(
            role="Resume Specialist",
            goal="Membuat resume singkat berisi poin-poin penting dari setiap topic.",
            backstory=(
                "Anda adalah peringkas profesional. "
                "Anda mengambil inti pembahasan dan menyajikannya dalam poin-poin penting."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        final_agent = Agent(
            role="Final Project Synthesizer",
            goal="Menyusun hasil akhir dari dua topic dalam satu project.",
            backstory=(
                "Anda adalah penyusun laporan akhir. "
                "Anda menggabungkan hasil pembahasan dua topic menjadi output final "
                "yang rapi, terstruktur, dan berbahasa Indonesia."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        quality_agent = Agent(
            role="Quality Assurance Agent",
            goal=(
                "Memeriksa kualitas output akhir agar sesuai dengan topic, jelas, lengkap, "
                "terstruktur, dan menggunakan bahasa Indonesia."
            ),
            backstory=(
                "Anda adalah pemeriksa kualitas output. "
                "Anda menilai apakah hasil akhir sudah relevan dengan dua topic, "
                "mudah dipahami, rapi, dan layak digunakan."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        base_context = ""
        if knowledge_base:
            base_context = f"\n\nKonteks tambahan:\n{knowledge_base}"

        main_task_1 = Task(
            description=(
                f"Project: {project_name}\n"
                f"Topic 1: {topic_1}\n\n"
                "Jelaskan topic ini dalam bahasa Indonesia. "
                "Bahas pengertian, manfaat, contoh penggunaan, dan kesimpulan singkat."
                f"{base_context}"
            ),
            expected_output=(
                "Jawaban utama untuk Topic 1 dalam bahasa Indonesia yang jelas, "
                "ringkas, dan sistematis."
            ),
            agent=research_agent,
        )

        review_task_1 = Task(
            description=(
                "Baca jawaban utama untuk Topic 1. "
                "Buat poin-poin review dalam bahasa Indonesia dengan format:\n"
                "1. Ringkasan jawaban utama\n"
                "2. Poin penting\n"
                "3. Kelebihan jawaban\n"
                "4. Catatan perbaikan\n"
                "5. Kesimpulan singkat"
            ),
            expected_output="Poin-poin review untuk Topic 1.",
            agent=review_agent,
            context=[main_task_1],
        )

        resume_task_1 = Task(
            description=(
                f"Buat resume akhir untuk Topic 1: {topic_1}. "
                "Fokus pada poin penting yang paling relevan. "
                "Gunakan bullet point dan bahasa Indonesia."
            ),
            expected_output="Resume penting untuk Topic 1.",
            agent=resume_agent,
            context=[main_task_1, review_task_1],
        )

        main_task_2 = Task(
            description=(
                f"Project: {project_name}\n"
                f"Topic 2: {topic_2}\n\n"
                "Jelaskan topic ini dalam bahasa Indonesia. "
                "Bahas pengertian, manfaat, contoh penggunaan, dan kesimpulan singkat."
                f"{base_context}"
            ),
            expected_output=(
                "Jawaban utama untuk Topic 2 dalam bahasa Indonesia yang jelas, "
                "ringkas, dan sistematis."
            ),
            agent=research_agent,
        )

        review_task_2 = Task(
            description=(
                "Baca jawaban utama untuk Topic 2. "
                "Buat poin-poin review dalam bahasa Indonesia dengan format:\n"
                "1. Ringkasan jawaban utama\n"
                "2. Poin penting\n"
                "3. Kelebihan jawaban\n"
                "4. Catatan perbaikan\n"
                "5. Kesimpulan singkat"
            ),
            expected_output="Poin-poin review untuk Topic 2.",
            agent=review_agent,
            context=[main_task_2],
        )

        resume_task_2 = Task(
            description=(
                f"Buat resume akhir untuk Topic 2: {topic_2}. "
                "Fokus pada poin penting yang paling relevan. "
                "Gunakan bullet point dan bahasa Indonesia."
            ),
            expected_output="Resume penting untuk Topic 2.",
            agent=resume_agent,
            context=[main_task_2, review_task_2],
        )

        final_task = Task(
            description=(
                f"Susun output final untuk project '{project_name}' berdasarkan dua topic berikut:\n"
                f"1. {topic_1}\n"
                f"2. {topic_2}\n\n"
                "Gunakan hasil jawaban utama, review, dan resume dari kedua topic. "
                "Output wajib berbahasa Indonesia dengan format:\n"
                "1. Nama project\n"
                "2. Daftar topic\n"
                "3. Ringkasan Topic 1\n"
                "4. Ringkasan Topic 2\n"
                "5. Perbandingan singkat kedua topic\n"
                "6. Poin penting gabungan\n"
                "7. Kesimpulan akhir"
            ),
            expected_output=(
                "Output final project dalam bahasa Indonesia yang memuat hasil dua topic "
                "secara ringkas dan terstruktur."
            ),
            agent=final_agent,
            context=[
                main_task_1,
                review_task_1,
                resume_task_1,
                main_task_2,
                review_task_2,
                resume_task_2,
            ],
        )

        quality_task = Task(
            description=(
                f"Periksa output final untuk project '{project_name}'. "
                "Buat laporan quality assurance dalam bahasa Indonesia dengan format:\n"
                "1. Kesesuaian dengan Topic 1 dan Topic 2\n"
                "2. Kejelasan isi\n"
                "3. Kelengkapan pembahasan\n"
                "4. Kerapian struktur\n"
                "5. Catatan perbaikan\n"
                "6. Keputusan akhir: Layak atau Perlu Revisi"
            ),
            expected_output=(
                "Laporan quality assurance dalam bahasa Indonesia yang menilai kualitas output final."
            ),
            agent=quality_agent,
            context=[final_task],
        )

        crew = Crew(
            agents=[
                research_agent,
                review_agent,
                resume_agent,
                final_agent,
                quality_agent,
            ],
            tasks=[
                main_task_1,
                review_task_1,
                resume_task_1,
                main_task_2,
                review_task_2,
                resume_task_2,
                final_task,
                quality_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        return {
            "status": "success",
            "project_name": project_name,
            "topics": [topic_1, topic_2],
            "agents": [
                agent_name,
                "Review Point Maker",
                "Resume Specialist",
                "Final Project Synthesizer",
                "Quality Assurance Agent",
            ],
            "model": model_to_use,
            "response": str(result),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }
