# ── UrbanAI ML API — Dockerfile ──────────────────────────────────────────────
FROM python:3.11-slim-bullseye

# Metadonnees
LABEL maintainer="mariem.negra@esprit.tn"
LABEL description="UrbanAI ML API v4 — DW dwurbanmobility SQL Server"
LABEL version="4.0.0"

# Variables environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Dossier de travail
WORKDIR /app

# Installer dependances systeme (ODBC Driver pour SQL Server)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list \
       > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements_api.txt .

# Installer dependances Python
RUN pip install --no-cache-dir -r requirements_api.txt

# Copier le code
COPY api_urbanai.py .

# Creer dossiers necessaires
RUN mkdir -p models logs

# Port expose
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Lancement
CMD ["uvicorn", "api_urbanai:app", "--host", "0.0.0.0", "--port", "8000"]
