# Gunakan image Python
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Buat directory kerja
WORKDIR /app

# Salin file requirements.txt dan instal dependensi
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Buat user non-root
RUN useradd -m appuser
USER appuser

# Salin semua file ke dalam container
COPY . /app/

# Jalankan aplikasi menggunakan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
