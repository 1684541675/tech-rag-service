ARG PYTHON_IMAGE=docker.m.daocloud.io/python:3.11-slim
FROM ${PYTHON_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai17_markdown_ingestion.py .
COPY ai18_jsonl_retrieval.py .
COPY ai19_retrieval_eval.py .
COPY ai20_glm_rag_answer.py .
COPY ai21_bagu_rag_api.py .
COPY 八股文.md .
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "ai21_bagu_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
