# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# Install project dependencies
RUN pip install --no-cache-dir ".[dev]"

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Create data and output directories
RUN mkdir -p data/raw data/sampled outputs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command: show help
ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["--help"]
