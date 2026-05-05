Gambaran Umum Project
Project ini dibuat untuk memproses data time series dan menghasilkan prediksi otomatis. Custom tool menjalankan Prophet, sedangkan agent CrewAI menafsirkan hasil prediksi menjadi report yang mudah dipahami.
Komponen	Fungsi
FastAPI	Menyediakan REST API untuk upload file dan cek status task.
Celery	Menjalankan proses forecast secara background.
Redis	Broker dan result backend Celery.
ProphetForecastTool	Custom tool untuk membaca data dan menjalankan Prophet().
CrewAI Agent	Membuat report prediksi berdasarkan hasil custom tool.
Ollama	Menjalankan local LLM untuk agent report.

Output	Excel, JSON, dan Markdown.Ambil path dari output_excel, output_json, atau markdown_file, lalu gunakan endpoint download.
GET http://127.0.0.1:8090/download?path=PATH_FILE

Output yang Dihasilkan
Format	Isi
Excel	Historical Data, Forecast All, Forecast Future, dan Summary.
JSON	Hasil forecast terstruktur untuk integrasi sistem.
Markdown	Report naratif dari agent CrewAI.



