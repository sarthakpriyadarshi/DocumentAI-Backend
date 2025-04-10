# agents/image_agent.py

from unstructured.partition.image import partition_image
from vector_store import add_chunks_to_vector_store
import os

def handle_image(file_path, session_id):
    # Partition the image file with unstructured.io.
    elements = partition_image(filename=file_path)
    # Build a list of chunks that preserves text and metadata.
    chunks = []
    for el in elements:
        if el.text:
            chunk_data = {
                "text": el.text,
                "source": os.path.basename(file_path),
                "type": el.__class__.__name__
            }
            # Optionally, if image elements have metadata, try to include it.
            if hasattr(el, "metadata"):
                try:
                    chunk_data.update(el.metadata.to_dict())
                except Exception as e:
                    print(f"WARNING: Could not extract metadata for image element: {e}")
            chunks.append(chunk_data)
    add_chunks_to_vector_store(session_id, chunks)
