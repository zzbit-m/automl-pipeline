FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.headless=true"]
