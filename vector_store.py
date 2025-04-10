import os
import re
import chromadb
from chromadb.utils import embedding_functions
from config import VECTOR_STORE_PATH
import config
import logging

logging.getLogger("chromadb").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
# Initialize ChromaDB Persistent Client and Collection with explicit HNSW config
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_or_create_collection(
    name="document_chunks",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
    metadata={"hnsw:space": "cosine"}  # Optional: explicit HNSW config for consistency
)

# Define a custom stop words set (you can adjust as needed)
stop_words = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "my", "myself",
    "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own",
    "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too",
    "under", "until", "up", "very",
    "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
])

def normalize_text(text):
    """
    Lowercase and remove punctuation from the text.
    """
    text = text.lower()
    return re.sub(r'[^\w\s]', '', text)

def add_chunks_to_vector_store(session_id: str, chunks: list):
    logger.info("[VECTOR STORE]: Adding Chunks to Vector Store Started")
    if all(isinstance(chunk, str) for chunk in chunks):
        chunks = [{"text": chunk, "source": "unknown", "type": "Unknown"} for chunk in chunks]
    
    documents = [chunk["text"] for chunk in chunks]
    metadatas = []
    last_title = None
    for i, chunk in enumerate(chunks):
        if chunk.get("type", "").lower() == "title":
            last_title = chunk["text"]
        metadatas.append({
            "source": chunk.get("source", "unknown"),
            "type": chunk.get("type", "Unknown"),
            "session_id": session_id,
            "position": i,
            "linked_title": last_title or "None"
        })
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    preview = f"{documents[:3]}, ..." if len(chunks) > 3 else str(documents[:3])
    logger.info("[VECTOR STORE]: Vector Stored Chunks Preview:", preview)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

def query_chunks(session_id: str, query: str, n_results: int = 5):
    logger.info("[VECTOR STORE]: Querying Chunks")
    logger.info(f"I[VECTOR STORE]: Session ID: {session_id}, Query: {query}, N Results: {n_results}")
    
    # Remove stop words from query
    all_words = query.lower().split()
    all_query_terms = [word for word in all_words if word not in stop_words]
    logger.info("[VECTOR STORE]: Query Terms after removing stop words:", all_query_terms)
    
    # Build a query string for embedding using only the filtered query terms
    query_for_embedding = ' '.join(all_query_terms)
    
    results = collection.query(
        query_texts=[query_for_embedding],
        n_results=n_results * 5,
        where={"session_id": session_id},
        include=["documents", "metadatas"]
    )
    
    # Build raw_chunks list from query results
    raw_chunks = [
        {
            "chunk": doc,
            "source": meta.get("source"),
            "type": meta.get("type"),
            "linked_title": meta.get("linked_title", ""),
            "position": meta.get("position", -1)
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    
    # Look for title chunks that match query terms and attempt to fetch the associated text chunk
    existing_positions = {chunk['position'] for chunk in raw_chunks}
    additional_chunks = []
    for chunk in raw_chunks:
        if chunk['type'].lower() == 'title' and any(term in normalize_text(chunk['chunk']) for term in all_query_terms):
            next_pos = chunk['position'] + 1
            if next_pos not in existing_positions:
                next_id = f"{session_id}_{next_pos}"
                next_result = collection.get(ids=[next_id])
                if next_result['documents']:
                    next_meta = next_result['metadatas'][0]
                    # If the next chunk is of type 'text' and its linked_title matches the current title, add it
                    if next_meta.get('type', '').lower() == 'text' and next_meta.get('linked_title') == chunk['chunk']:
                        additional_chunks.append({
                            'chunk': next_result['documents'][0],
                            'source': next_meta.get('source'),
                            'type': next_meta.get('type'),
                            'linked_title': next_meta.get('linked_title'),
                            'position': next_meta.get('position')
                        })
                        existing_positions.add(next_pos)
    
    # Combine raw_chunks and additional_chunks
    all_chunks = raw_chunks + additional_chunks
    
    def relevance_score(chunk):
        score = 0
        linked_title_lower = normalize_text(str(chunk.get('linked_title', '')))
        chunk_text_lower = normalize_text(str(chunk.get('chunk', '')))
        chunk_type_lower = str(chunk.get('type', '')).lower()
        position = chunk.get('position', -1)
        if any(term in linked_title_lower for term in all_query_terms):
            score += 4
        if any(term in chunk_text_lower for term in all_query_terms):
            score += 2
        if chunk_type_lower == 'text' and any(term in linked_title_lower for term in all_query_terms):
            score += 4
        if position > 0:
            score -= (position - 1) * 0.1
        return score

    sorted_chunks = sorted(all_chunks, key=relevance_score, reverse=True)
    
    # Apply merging: if a chunk is of type 'title', merge it with the next chunk if that next chunk is 'text'
    merged_results = []
    seen_positions = set()
    for chunk in sorted_chunks:
        pos = chunk['position']
        if pos in seen_positions:
            continue
        if chunk['type'].lower() == 'title':
            next_chunk_candidate = next((c for c in all_chunks if c['position'] == pos + 1 and c['type'].lower() == 'text'), None)
            if next_chunk_candidate and next_chunk_candidate['linked_title'] == chunk['chunk']:
                merged_chunk = {
                    'chunk': f"{chunk['chunk']} {next_chunk_candidate['chunk']}",
                    'source': chunk['source'],
                    'type': 'Merged Title+Text',
                    'linked_title': chunk['linked_title'],
                    'position': pos
                }
                merged_results.append(merged_chunk)
                seen_positions.add(pos)
                seen_positions.add(pos + 1)
                continue
        merged_results.append(chunk)
        seen_positions.add(pos)

    # Select top n_results from merged_results
    top_chunks = merged_results[:n_results]
    
    logger.info("[VECTOR STORE]: Top Chunks:")
    for idx, chunk in enumerate(top_chunks):
        score = relevance_score(chunk) if chunk.get('type') != 'Merged Title+Text' else "Merged"
        logger.info(f"[VECTOR STORE]: Rank {idx+1}: Chunk: {chunk['chunk'][:50]}..., Type: {chunk['type']}, Position: {chunk['position']}, Score: {score}")
    
    return top_chunks

def sanitize_metadata(metadata: dict) -> dict:
    """
    Recursively convert non-basic types in metadata to string representations.
    """
    def is_valid_type(value):
        return isinstance(value, (str, int, float, bool))
    sanitized = {}
    for key, value in metadata.items():
        if is_valid_type(value):
            sanitized[key] = value
        else:
            try:
                sanitized[key] = str(value)
            except Exception:
                sanitized[key] = "UNSERIALIZABLE"
    return sanitized

def get_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    from langchain_chroma import Chroma
    return Chroma(
        collection_name="document_chunks",
        embedding_function=embeddings,
        persist_directory=config.VECTOR_STORE_PATH
    )
