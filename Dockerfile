FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /nb-whisper-api

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765" ]