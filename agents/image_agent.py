import os
import logging
from unstructured.partition.image import partition_image

app_logger = logging.getLogger("DocumentAI")
app_logger.setLevel(logging.INFO)

def handle_image(file_path, session_id):
    app_logger.info("Started Image Agent:")
    
    # Partition the image file
    elements = partition_image(filename=file_path)
    
    # Build chunks as a list of dictionaries
    chunks = []
    for el in elements:
        if hasattr(el, "text") and el.text:
            chunk_data = {
                "text": el.text,
                "source": os.path.basename(file_path),
                "type": el.__class__.__name__
            }
            # Handle metadata safely
            if hasattr(el, "metadata"):
                try:
                    metadata = el.metadata.to_dict()
                    # Sanitize metadata: convert non-scalar values to strings
                    for key, value in metadata.items():
                        if not isinstance(value, (str, int, float, bool)):
                            metadata[key] = str(value)
                    chunk_data.update(metadata)
                except Exception as e:
                    app_logger.warning("Could not extract metadata for image element: %s", e)
            chunks.append(chunk_data)
    
    app_logger.info("[Image Agent]: Partitioned Image Elements: %s", chunks[:1])
    return chunks, session_id