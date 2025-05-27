"""
Portfolio RAG API

This module implements a FastAPI-based Retrieval Augmented Generation (RAG) system
for portfolio data analysis. It provides endpoints for document upload, semantic search,
and enhanced chatbot interaction with streaming support.

The API uses FAISS for efficient vector similarity search and Sentence Transformers
for generating text embeddings.

Example:
    To run the API:
        $ uvicorn rag.main:app --reload

    The API will be available at:
        - http://localhost:8000/docs (Swagger UI)
        - http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from rag.utils.embeddings import get_embeddings
from rag.utils.vector_store import add_to_vector_store, search_vector_store
from rag.utils.file_processor import process_file
from rag.utils.enhanced_chatbot import EnhancedChatbot
import os
import json
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio RAG API",
    description="API for Retrieval Augmented Generation on Portfolio Data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for LLM function - replace with actual LLM implementation
async def llm_fn(prompt: str) -> str:
    """Placeholder LLM function that returns a simple response."""
    return f"This is a placeholder response for: {prompt}"

# Initialize enhanced chatbot
chatbot = EnhancedChatbot(
    get_embeddings_fn=get_embeddings,
    search_vector_store_fn=search_vector_store,
    llm_fn=llm_fn
)

class QueryRequest(BaseModel):
    """
    Request model for the query endpoint.
    
    Attributes:
        query (str): The natural language query to search for
        top_k (int, optional): Number of results to return. Defaults to 5
    """
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    """
    Response model for the query endpoint.
    
    Attributes:
        matches (List[dict]): List of matching documents with metadata
        query_embedding (List[float]): Vector representation of the query
    """
    matches: List[dict]
    query_embedding: List[float]

class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.
    
    Attributes:
        message (str): The user's message
        session_id (str, optional): Session identifier for conversation continuity
    """
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint.
    
    Attributes:
        response (str): The chatbot's response
        session_id (str): The session identifier
        conversation_history (List[Dict[str, Any]]): The current conversation history
    """
    response: str
    session_id: str
    conversation_history: List[Dict[str, Any]]

@app.post("/upload", tags=["Document Management"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document for RAG.
    
    This endpoint handles document upload, processing, and indexing. It supports
    multiple file formats including PDF, DOCX, TXT, and Excel files.
    
    Args:
        file (UploadFile): The file to be processed
        
    Returns:
        dict: A message indicating success and the number of chunks processed
        
    Raises:
        HTTPException: If file processing or indexing fails
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/upload" \\
             -H "Content-Type: multipart/form-data" \\
             -F "file=@document.pdf"
        ```
    """
    try:
        # Create upload directory if it doesn't exist
        os.makedirs("upload", exist_ok=True)
        
        # Save file temporarily
        file_path = f"upload/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file and get text chunks
        chunks = process_file(file_path)
        
        # Get embeddings for chunks
        embeddings = get_embeddings(chunks)
        
        # Add to vector store
        add_to_vector_store(chunks, embeddings, metadata={"source": file.filename})
        
        # Clean up
        os.remove(file_path)
        
        return {"message": f"Successfully processed {file.filename}", "chunks": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def query(request: QueryRequest):
    """
    Query the RAG system with natural language.
    
    This endpoint performs semantic search over the processed documents using
    the query text. It returns the most relevant matches along with their
    similarity scores.
    
    Args:
        request (QueryRequest): The query request containing the search text
            and number of results to return
        
    Returns:
        QueryResponse: The matches and query embedding
        
    Raises:
        HTTPException: If query processing fails
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/query" \\
             -H "Content-Type: application/json" \\
             -d '{"query": "What are the top holdings?", "top_k": 5}'
        ```
    """
    try:
        # Get query embedding
        query_embedding = get_embeddings([request.query])[0]
        
        # Search vector store
        matches = search_vector_store(query_embedding, k=request.top_k)
        
        return {
            "matches": matches,
            "query_embedding": query_embedding.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the enhanced RAG-powered chatbot.
    
    This endpoint provides a streaming interface to the chatbot. The response
    is streamed back to the client as it's generated, with proper handling
    of conversation history and session management.
    
    Args:
        request (ChatRequest): The chat request containing the user's message
            and optional session ID
        
    Returns:
        StreamingResponse: A streaming response containing the generated text
        
    Raises:
        HTTPException: If chat processing fails
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/chat" \\
             -H "Content-Type: application/json" \\
             -d '{"message": "What are the top holdings?", "session_id": "optional-session-id"}'
        ```
        
    Note:
        The response is streamed as newline-delimited JSON objects, where each
        object contains a 'token' field with the next piece of generated text.
    """
    try:
        async def generate():
            async for token in chatbot.get_streaming_response(
                request.message,
                request.session_id
            ):
                yield json.dumps({"token": token}) + "\n"
        
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset", tags=["Chat"])
async def reset_chat(session_id: str):
    """
    Reset a chat session.
    
    This endpoint clears the conversation history and memory for a specific
    session. If the session doesn't exist, returns an error.
    
    Args:
        session_id (str): The session to reset
        
    Returns:
        dict: A success message
        
    Raises:
        HTTPException: If the session doesn't exist
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/chat/reset?session_id=your-session-id"
        ```
    """
    try:
        if chatbot.reset_session(session_id):
            return {"message": "Session reset successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Check the health status of the API.
    
    This endpoint provides a simple way to verify that the API is running
    and responding to requests.
    
    Returns:
        dict: A status message indicating the API is healthy
        
    Example:
        ```bash
        curl -X GET "http://localhost:8000/health"
        ```
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 