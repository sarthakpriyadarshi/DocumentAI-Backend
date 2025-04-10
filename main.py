# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging

logging.basicConfig(level=logging.INFO)
from processing import get_file_type
from session import create_session, is_valid_session
# Instead of the old ask_ollama, use the new LLM instance from llm.py
from llm import ollama_llm
from vector_store import add_chunks_to_vector_store, get_vectorstore, query_chunks
# Assuming you have agent modules for different file types
from agents import document_agent, image_agent, audio_agent

from langchain.prompts import PromptTemplate

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

class AskRequest(BaseModel):
    session_id: str
    question: str

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following context to answer the question.\n\n{context}\n\nQ: {question}\nA:"
)


app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Create a new session
    session_id = create_session()

    # Determine file type using the processing utility
    file_type = get_file_type(temp_path)
    if file_type == "document":
        # Process file to extract chunks and index into vector store.
        chunks, session_id = document_agent.handle_document(temp_path, session_id)
        # Now add the chunks to the LangChain vector store.
        add_chunks_to_vector_store(session_id, chunks)
    elif file_type == "image":
        image_agent.handle_image(temp_path, session_id)
    elif file_type == "audio":
        audio_agent.handle_audio(temp_path, session_id)
    else:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    os.remove(temp_path)
    return {"session_id": session_id}

# Build the RetrievalQA chain using the new Ollama LLM and LangChain Chroma.
# Instantiate the vector store once.
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following context to answer the question.\n\n{context}\n\nQ: {question}\nA:"
)



# qa_chain = LLMChain(llm=ollama_llm, prompt=prompt)

qa_chain = prompt | ollama_llm

@app.post("/ask")
async def ask_question(request: AskRequest):
    session_id = request.session_id
    question = request.question

    if not is_valid_session(session_id):
        return JSONResponse({"error": "Invalid or expired session"}, status_code=400)
    
    try:
        # Retrieve the relevant chunks and build the context string:
        top_chunks = query_chunks(session_id=session_id, query=question)
        context = "\n\n".join([chunk["chunk"] for chunk in top_chunks if "chunk" in chunk])

        if not context:
            return JSONResponse({"answer": "No relevant context found."})
        
        # Now call the QA chain with a dictionary containing both keys.
        answer = await qa_chain.ainvoke({"context": context, "question": question})
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
