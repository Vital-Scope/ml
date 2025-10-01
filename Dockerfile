FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Зависимости
COPY . .
RUN pip install --no-cache-dir -r ./src/requirements.txt

# Код приложения


# Директория для временных файлов
RUN mkdir -p /app/data

# Порт сервиса
EXPOSE 8082

# Рабочая директория и команда запуска
WORKDIR /app
USER root
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
CMD ["python", "run.py"]