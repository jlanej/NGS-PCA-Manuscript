FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/jlanej/NGS-PCA-Manuscript"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY run_all.sh .
RUN chmod +x run_all.sh

# Default data / output mounts
ENV NGSPCA_DATA_DIR=/data \
    NGSPCA_OUTPUT_DIR=/output \
    NGSPCA_SUBSET=0

CMD ["./run_all.sh"]
