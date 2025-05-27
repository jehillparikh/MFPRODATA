"""
Text Embedding Generation Module

This module handles the generation of text embeddings using the Sentence Transformers
library. It uses the 'all-MiniLM-L6-v2' model, which is optimized for semantic
similarity tasks and provides a good balance between performance and accuracy.

The embeddings are 384-dimensional vectors that capture the semantic meaning of
the input text, making them suitable for similarity search and retrieval tasks.

Example:
    ```python
    from rag.utils.embeddings import get_embeddings

    texts = ["Sample text 1", "Sample text 2"]
    embeddings = get_embeddings(texts)
    ```
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using SentenceTransformers.
    
    This function processes a list of text strings and returns their vector
    representations (embeddings) using the all-MiniLM-L6-v2 model. The
    embeddings are normalized and can be used for semantic similarity search.
    
    Args:
        texts (List[str]): List of text strings to generate embeddings for.
            Each text string can be of any length, but very long texts may
            be truncated by the model.
        
    Returns:
        np.ndarray: Array of embeddings with shape (n_texts, 384) where
            n_texts is the number of input texts and 384 is the embedding
            dimension.
        
    Raises:
        Exception: If there's an error during embedding generation, with
            details about what went wrong.
        
    Example:
        ```python
        texts = [
            "Portfolio includes tech stocks",
            "High yield bonds in portfolio"
        ]
        embeddings = get_embeddings(texts)
        # embeddings.shape = (2, 384)
        ```
        
    Note:
        The model uses a maximum sequence length of 256 tokens. Longer
        texts will be truncated to this length.
    """
    try:
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_tensor=True)
        
        # Convert to numpy array
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
            
        return embeddings
    
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise 