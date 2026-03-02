from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
from itertools import cycle
import asyncio, shutil, os,tempfile, torch


app = FastAPI()

# Reserves 2 spots on VRAM
instanceNumber = 2

# Pipeline
models = [
    pipeline("automatic-speech-recognition", model= "NbAiLab/nb-whisper-large", dtype=torch.float16,device="cuda",
             chunk_length_s=28,ignore_warning=True,generate_kwargs={'task': 'transcribe', 'language': 'no', "num_beams":1} )
             # Creates more instances for holding queue
             for _ in range(instanceNumber)
]

modelPool = cycle(models)

poolLock = asyncio.Semaphore(instanceNumber)

@app.post("/v1/audio/transcriptions/")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default ="nb-whisper-large")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmpPath = tmp.name
    try: 
        async with poolLock:
            asr = next(modelPool)
            result = await asyncio.to_thread(asr, tmpPath)
    finally:
        os.unlink(tmpPath)
    
    return {"text": result["text"].strip()}

# Basic helth check
@app.get("/health")
def health_check():
    return {"status" : "healthy"}