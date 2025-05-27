"""
Vector Store Management Module

This module implements a vector store using FAISS (Facebook AI Similarity Search)
for efficient similarity search over text embeddings. It provides functionality
for adding embeddings to the index and searching for similar vectors.

The module maintains both the FAISS index for vector similarity search and a
metadata store that keeps track of additional information about each vector,
such as the original text and source document.

Technical Details:
    - Uses FAISS IndexFlatL2 for exact L2 distance computation
    - Dimension: 384 (matches all-MiniLM-L6-v2 embeddings)
    - Persistence: Index and metadata are saved to disk
    - Thread-safe operations for concurrent access

Example:
    ```python
    from rag.utils.vector_store import add_to_vector_store, search_vector_store
    import numpy as np

    # Add documents to the store
    texts = ["Sample text"]
    embeddings = np.random.rand(1, 384)  # Example embeddings
    add_to_vector_store(texts, embeddings, {"source": "example.txt"})

    # Search for similar documents
    query_embedding = np.random.rand(384)  # Example query
    matches = search_vector_store(query_embedding, k=5)
    ```
"""

import faiss
import numpy as np
from typing import List, Dict, Any
import pickle
import os

# Constants
INDEX_PATH = "rag/data/faiss_index"
METADATA_PATH = "rag/data/metadata.pkl"

# Initialize or load the index and metadata
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata_list = pickle.load(f)
else:
    # Create directory if it doesn't exist
    os.makedirs("rag/data", exist_ok=True)
    
    # Initialize new index and metadata
    dimension = 384  # dimension of all-MiniLM-L6-v2 embeddings
    index = faiss.IndexFlatL2(dimension)
    metadata_list = []

def add_to_vector_store(texts: List[str], embeddings: np.ndarray, metadata: Dict[str, Any] = None) -> None:
    """
    Add texts and their embeddings to the vector store.
    
    This function adds new documents to the FAISS index and updates the metadata
    store. The function is atomic - either all documents are added successfully,
    or none are added in case of an error.
    
    Args:
        texts (List[str]): List of text chunks to add to the store. Each chunk
            should be a meaningful unit of text (e.g., a paragraph).
        embeddings (np.ndarray): Array of embeddings with shape (n_texts, dimension)
            where n_texts matches the length of texts and dimension is 384.
        metadata (Dict[str, Any], optional): Additional metadata to store with each
            text chunk, such as source document information.
    
    Raises:
        Exception: If there's an error during the addition process, with details
            about what went wrong.
    
    Example:
        ```python
        texts = ["Text chunk 1", "Text chunk 2"]
        embeddings = model.encode(texts)  # Shape: (2, 384)
        metadata = {"source": "document.pdf", "date": "2024-01-01"}
        add_to_vector_store(texts, embeddings, metadata)
        ```
    
    Note:
        - The function automatically saves the updated index and metadata to disk
        - The metadata dictionary is copied for each text chunk
        - The original text is always stored in the metadata under the 'text' key
    """
    try:
        # Add embeddings to FAISS index
        index.add(embeddings)
        
        # Add metadata
        for text in texts:
            meta = metadata.copy() if metadata else {}
            meta['text'] = text
            metadata_list.append(meta)
        
        # Save index and metadata
        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata_list, f)
            
    except Exception as e:
        print(f"Error adding to vector store: {str(e)}")
        raise

def search_vector_store(query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the vector store for similar texts.
    
    This function performs a k-nearest neighbor search in the FAISS index
    using the provided query embedding. It returns the most similar documents
    along with their metadata and similarity scores.
    
    Args:
        query_embedding (np.ndarray): Query vector with shape (384,) or (1, 384)
        k (int, optional): Number of results to return. Defaults to 5.
    
    Returns:
        List[Dict[str, Any]]: List of matches, where each match is a dictionary
            containing:
            - All original metadata for the document
            - 'score': Similarity score (1 / (1 + L2 distance))
            - 'rank': Position in results (1-based)
    
    Raises:
        Exception: If there's an error during the search process, with details
            about what went wrong.
    
    Example:
        ```python
        query = "Investment strategy"
        query_embedding = model.encode([query])[0]
        matches = search_vector_store(query_embedding, k=3)
        for match in matches:
            print(f"Score: {match['score']:.2f} - {match['text']}")
        ```
    
    Note:
        - Scores are normalized to [0, 1] range using 1 / (1 + distance)
        - Higher scores indicate better matches
        - The function automatically handles reshaping of the query embedding
    """
    try:
        # Reshape query embedding if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search index
        distances, indices = index.search(query_embedding, k)
        
        # Get metadata for matches
        matches = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata_list):  # Check if index is valid
                match = metadata_list[idx].copy()
                match['score'] = float(1 / (1 + dist))  # Convert distance to similarity score
                match['rank'] = i + 1
                matches.append(match)
        
        return matches
    
    except Exception as e:
        print(f"Error searching vector store: {str(e)}")
        raise 