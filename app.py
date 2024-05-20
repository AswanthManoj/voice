import shutil
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, Dict, List
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess, time, os, requests, asyncio
from fastapi import FastAPI, File, UploadFile, Request
from voice import TextToSpeech, SpeechToText, LanguageModel
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TTS = TextToSpeech()
STT = SpeechToText()
LLM = LanguageModel()

audio_dir = 'audio'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

def iterfile(stream_file):
    with open(stream_file, mode='rb') as file_like:
        yield from file_like

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_audio", response_class=StreamingResponse)
async def process_audio(audio_file: UploadFile = File(...)):
        
    transcript = await STT.listen(audio_file.file)
    async for response in LLM.respond(transcript):
        print(response, end="", flush=True)
    output_audio_path = await TTS.speak(response)

    return StreamingResponse(iterfile(output_audio_path), media_type='audio/*')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)