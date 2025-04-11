# agents/document_agent.py

import os
import logging
from unstructured.partition.auto import partition

# Use the custom app logger
app_logger = logging.getLogger("DocumentAI")
app_logger.setLevel(logging.INFO)

# Suppress verbose logs from dependencies
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pikepdf").setLevel(logging.ERROR)

def handle_document(file_path, session_id):
    app_logger.info("Started Document Agent:")

    # Partition the document with unstructured.io; this returns a list of element objects.
    elements = partition(filename=file_path)

    # Build a list of dictionaries that preserves the text, element type, and all metadata.
    chunks = []
    for el in elements:
        # Ensure the element has text and it is non-empty.
        if hasattr(el, "text") and el.text:
            # Start with basic metadata: source and type.
            chunk_data = {
                "text": el.text,
                "source": os.path.basename(file_path),
                "type": el.__class__.__name__
            }
            # If there is a metadata attribute, attempt to add its dictionary representation.
            if hasattr(el, "metadata"):
                try:
                    # Merge the metadata into the chunk dictionary.
                    chunk_data.update(el.metadata.to_dict())
                except Exception as e:
                    app_logger.warning("Could not extract metadata for element: %s", e)
            chunks.append(chunk_data)
    
    # Print a preview of the resulting chunks.
    if len(chunks) > 2:
        preview = f"{chunks[:1]}, ..."
    else:
        preview = str(chunks[:10])
    app_logger.info("[Document Agent]: Partitioned Document Agent Elements: %s", preview)
    
    # Send the full chunks with metadata to the vector store.
    return chunks, session_id
