FROM python:3.12-slim

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Build React app ───────────────────────────────────────────────────────────
COPY webapp/package.json webapp/package-lock.json ./webapp/
RUN cd webapp && npm ci --silent
COPY webapp/ ./webapp/
RUN cd webapp && npm run build

# ── Copy application code ─────────────────────────────────────────────────────
COPY api/      ./api/
COPY results/  ./results/

# ── Runtime dirs ─────────────────────────────────────────────────────────────
RUN mkdir -p models \
    results/runs/classify/yolo_metrics/yolo_v2_scratch/weights

# ── HF Spaces runs on port 7860 ───────────────────────────────────────────────
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
