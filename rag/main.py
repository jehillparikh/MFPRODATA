"""
Portfolio RAG API

This module implements a FastAPI-based Retrieval Augmented Generation (RAG) system
for portfolio data analysis. It provides endpoints for document upload, semantic search,
and health monitoring.

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
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag.utils.embeddings import get_embeddings
from rag.utils.vector_store import add_to_vector_store, search_vector_store
from rag.utils.file_processor import process_file
import os

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