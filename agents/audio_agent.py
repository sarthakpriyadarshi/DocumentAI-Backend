# agents/audio_agent.py

import whisper
from vector_store import add_chunks_to_vector_store

model = whisper.load_model("base")  # Local model

def handle_audio(file_path, session_id):
    result = model.transcribe(file_path)
    text = result["text"]
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    add_chunks_to_vector_store(session_id, chunks)
