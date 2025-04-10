import uuid
import time

sessions = {}

def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = time.time()
    return session_id

def is_valid_session(session_id):
    return session_id in sessions and time.time() - sessions[session_id] < 3600
