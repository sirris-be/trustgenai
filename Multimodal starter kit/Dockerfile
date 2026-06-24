FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir --upgrade pip

COPY . /app

RUN pip install --no-cache-dir -e .

CMD sh -c "fastapi run tgenai_agent/fair_server.py --host 0.0.0.0 --port ${AGENT_PORT}"