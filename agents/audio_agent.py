# agents/audio_agent.py

import os
import whisper
import logging

# Configure logging
app_logger = logging.getLogger("DocumentAI")
app_logger.setLevel(logging.INFO)

# Load the Whisper model
model = whisper.load_model("base")  # Local model

def handle_audio(file_path, session_id):
    """Transcribe audio and return chunks as a list of dictionaries.

    Args:
        file_path (str): Path to the audio file to process.
        session_id (str): Unique identifier for the session.

    Returns:
        tuple: (session_id, formatted_chunks) where formatted_chunks is a list of dictionaries
               containing 'text', 'source', and 'type' keys.
    """
    app_logger.info("[Audio Agent]: Processing audio file: %s", file_path)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        app_logger.error("[Audio Agent]: File not found: %s", file_path)
        return session_id, []
    
    # Transcribe the audio file with error handling
    try:
        result = model.transcribe(file_path)
        text = result["text"]
        app_logger.info("[Audio Agent]: Transcription successful for %s", file_path)
    except Exception as e:
        app_logger.error("[Audio Agent]: Transcription failed for %s: %s", file_path, str(e))
        return session_id, []
    
    # Handle empty transcription
    if not text.strip():
        app_logger.warning("[Audio Agent]: No text transcribed from %s", file_path)
        return session_id, []
    
    # Split text into chunks of 500 characters
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    
    # Format chunks as a list of dictionaries, matching the expected structure
    formatted_chunks = [
        {
            "text": chunk.strip(),  # Remove leading/trailing whitespace
            "source": os.path.basename(file_path),  # Use filename as source
            "type": "AudioText"  # Custom type for audio chunks
        }
        for chunk in chunks if chunk.strip()  # Skip empty chunks
    ]
    
    app_logger.info("[Audio Agent]: Generated %d chunks for %s", len(formatted_chunks), file_path)
    return session_id, formatted_chunks