import mimetypes

def get_file_type(file_path: str):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith("image"):
            return "image"
        elif mime_type.startswith("audio"):
            return "audio"
        elif mime_type.startswith("text") or file_path.endswith(".pdf") or file_path.endswith(".docx"):
            return "document"
    return "unknown"
