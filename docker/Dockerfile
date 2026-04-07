FROM pytorch/pytorch:2.0.1-cuda11.7-runtime
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
ENTRYPOINT ["python", "-m", "src.cli"]
