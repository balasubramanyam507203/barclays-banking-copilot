FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY backend/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        -r /tmp/requirements.txt


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"
ENV HF_HOME="/home/appuser/.cache/huggingface"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 appgroup \
    && useradd \
        --system \
        --uid 10001 \
        --gid appgroup \
        --create-home \
        --home-dir /home/appuser \
        appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace/backend

COPY backend/ /workspace/backend/
COPY sample_documents/ /workspace/sample_documents/
COPY evaluation/ /workspace/evaluation/

RUN chmod +x /workspace/backend/docker-entrypoint.sh \
    && mkdir -p \
        /workspace/backend/local_indexes \
        /workspace/backend/local_data \
        /home/appuser/.cache/huggingface \
    && chown -R appuser:appgroup \
        /workspace \
        /home/appuser

USER appuser

EXPOSE 8000

ENTRYPOINT ["/workspace/backend/docker-entrypoint.sh"]
