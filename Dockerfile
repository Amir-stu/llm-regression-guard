# Multi-stage so the runtime image carries no build toolchain.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install ".[openai]"

FROM python:3.11-slim

# Never run the eval as root: it executes model output through a JSON parser
# and writes reports, neither of which needs privilege.
RUN useradd --create-home --uid 10001 evaluser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY data/ data/
COPY prompts/ prompts/

# Run history and reports are the only mutable state. Mount a volume here to
# keep the baseline across container restarts - without it, every run reports
# "no baseline" and passes unconditionally.
RUN mkdir -p /app/runs && chown -R evaluser:evaluser /app
VOLUME ["/app/runs"]

USER evaluser

ENV RG_PROVIDER=mock     RG_DATASET=/app/data/golden_dataset_v1.json     RG_PROMPT_DIR=/app/prompts     RG_DB_PATH=/app/runs/eval_runs.db     RG_REPORT_DIR=/app/runs/reports     PYTHONUNBUFFERED=1

# Fails the health check if the dataset or prompts are unreadable.
HEALTHCHECK --interval=30s --timeout=10s --retries=2 CMD ["rg", "validate"]

ENTRYPOINT ["rg"]
CMD ["run", "--prompt", "latest"]
