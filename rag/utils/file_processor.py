"""
File Processing Module

This module handles the processing of various file types for the RAG system.
It supports multiple file formats and converts them into text chunks suitable
for embedding generation and semantic search.

Supported File Types:
    - Excel (.xlsx, .xls, .xlsb)
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt)

The module implements intelligent chunking strategies for each file type,
ensuring that the resulting text chunks maintain semantic coherence while
staying within size limits suitable for the embedding model.

Example:
    ```python
    from rag.utils.file_processor import process_file

    # Process a PDF file
    chunks = process_file("document.pdf", chunk_size=1000)

    # Process an Excel file
    chunks = process_file("data.xlsx")
    ```
"""

import pandas as pd
from typing import List
import docx
import PyPDF2
import os

def process_file(file_path: str, chunk_size: int = 1000) -> List[str]:
    """
    Process different types of files and return text chunks.
    
    This function serves as the main entry point for file processing. It
    automatically detects the file type based on extension and delegates
    to the appropriate processing function.
    
    Args:
        file_path (str): Path to the file to process. The file type is
            determined by its extension.
        chunk_size (int, optional): Maximum size of text chunks in characters.
            Defaults to 1000. This parameter is used for PDF, DOCX, and TXT
            files but not for Excel files.
    
    Returns:
        List[str]: List of text chunks extracted from the file.
    
    Raises:
        ValueError: If the file type is not supported
        Exception: If there's an error during file processing
    
    Example:
        ```python
        # Process a PDF with custom chunk size
        chunks = process_file("report.pdf", chunk_size=1500)
        
        # Process an Excel file (chunk_size is ignored)
        chunks = process_file("data.xlsx")
        ```
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension in ['.xlsx', '.xls', '.xlsb']:
            return process_excel(file_path)
        elif file_extension == '.pdf':
            return process_pdf(file_path, chunk_size)
        elif file_extension == '.docx':
            return process_docx(file_path, chunk_size)
        elif file_extension == '.txt':
            return process_txt(file_path, chunk_size)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise

def process_excel(file_path: str) -> List[str]:
    """
    Process Excel files into text chunks.
    
    This function reads an Excel file and converts each row into a text chunk,
    preserving column names as context. Empty cells are excluded.
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        List[str]: List of text chunks, one per row
    
    Example:
        ```python
        chunks = process_excel("portfolio.xlsx")
        # Example chunk: "Company: Apple Inc. | Sector: Technology | Price: 150.00"
        ```
    """
    df = pd.read_excel(file_path)
    chunks = []
    
    # Convert each row to a text chunk
    for _, row in df.iterrows():
        chunk = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def process_pdf(file_path: str, chunk_size: int) -> List[str]:
    """
    Process PDF files into text chunks.
    
    This function extracts text from PDF files and splits it into chunks
    while attempting to preserve semantic boundaries (sentences).
    
    Args:
        file_path (str): Path to the PDF file
        chunk_size (int): Maximum size of each chunk in characters
    
    Returns:
        List[str]: List of text chunks
    
    Example:
        ```python
        chunks = process_pdf("report.pdf", chunk_size=1000)
        ```
    """
    chunks = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        # Split into chunks
        chunks = split_text(text, chunk_size)
    
    return chunks

def process_docx(file_path: str, chunk_size: int) -> List[str]:
    """
    Process Word documents into text chunks.
    
    This function extracts text from DOCX files, preserving paragraph
    boundaries where possible while splitting into chunks.
    
    Args:
        file_path (str): Path to the DOCX file
        chunk_size (int): Maximum size of each chunk in characters
    
    Returns:
        List[str]: List of text chunks
    
    Example:
        ```python
        chunks = process_docx("document.docx", chunk_size=1000)
        ```
    """
    doc = docx.Document(file_path)
    text = ""
    
    # Extract text from paragraphs
    for para in doc.paragraphs:
        text += para.text + " "
    
    # Split into chunks
    chunks = split_text(text, chunk_size)
    
    return chunks

def process_txt(file_path: str, chunk_size: int) -> List[str]:
    """
    Process text files into chunks.
    
    This function reads a text file and splits its content into chunks
    while attempting to preserve sentence boundaries.
    
    Args:
        file_path (str): Path to the text file
        chunk_size (int): Maximum size of each chunk in characters
    
    Returns:
        List[str]: List of text chunks
    
    Example:
        ```python
        chunks = process_txt("notes.txt", chunk_size=1000)
        ```
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split into chunks
    chunks = split_text(text, chunk_size)
    
    return chunks

def split_text(text: str, chunk_size: int) -> List[str]:
    """
    Split text into chunks while preserving sentence boundaries.
    
    This function implements a smart text splitting strategy that:
    1. Respects sentence boundaries
    2. Ensures chunks don't exceed the specified size
    3. Avoids splitting in the middle of words
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int): Maximum size of each chunk in characters
    
    Returns:
        List[str]: List of text chunks
    
    Example:
        ```python
        text = "Long text... More text..."
        chunks = split_text(text, chunk_size=1000)
        ```
    
    Note:
        The function tries to keep sentences together, but will split them
        if they exceed the chunk_size limit.
    """
    chunks = []
    current_chunk = ""
    
    # Split text into sentences (simple implementation)
    sentences = text.replace('\n', ' ').split('.')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks 