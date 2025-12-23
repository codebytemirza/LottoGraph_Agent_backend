# ============================================================================
# Multi-stage Dockerfile for FastAPI + LangGraph Production
# Optimized for Coolify deployment with security and performance best practices
# ============================================================================

# ============================================================================
# Stage 1: Builder - Compile dependencies
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies (MariaDB replaces MySQL in Debian Trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libmariadb-dev \
    libmariadb-dev-compat \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install pip tools and create wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential runtime dependencies (MariaDB client instead of MySQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libmariadb3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -s /sbin/nologin appuser

# Copy wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install dependencies from pre-built wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels /root/.cache /tmp/*

# Copy application code with correct ownership
COPY --chown=appuser:appuser . .

# Create logs directory with correct permissions
RUN mkdir -p logs && chown -R appuser:appuser logs

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 3000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=3000 \
    HOST=0.0.0.0

# Health check (curl must be available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run with uvicorn (single worker for stability, Coolify handles scaling)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--log-level", "info"]