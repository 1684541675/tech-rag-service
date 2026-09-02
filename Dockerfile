ARG PYTHON_IMAGE=docker.m.daocloud.io/python:3.11-slim
FROM ${PYTHON_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "scripts.serve_ai26_agent:app", "--host", "0.0.0.0", "--port", "8000"]
