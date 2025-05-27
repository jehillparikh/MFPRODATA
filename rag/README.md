# Portfolio RAG API

A Retrieval Augmented Generation (RAG) API for portfolio data analysis using FastAPI, FAISS, and Sentence Transformers.

## Features

- 📄 Multi-format document support (PDF, DOCX, TXT, Excel)
- 🔍 Semantic search using FAISS vector store
- 🚀 Fast and efficient retrieval
- 🔄 Real-time document processing
- 📊 Support for portfolio data analysis

## Architecture

```
rag/
├── main.py                 # FastAPI application
├── utils/
│   ├── embeddings.py      # Text embedding generation
│   ├── vector_store.py    # FAISS vector store management
│   └── file_processor.py  # File processing utilities
├── data/                  # Directory for storing FAISS index and metadata
└── README.md             # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

Run the FastAPI server:
```bash
uvicorn rag.main:app --reload
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

### API Endpoints

1. **Upload Document** (`POST /upload`)
   - Upload and process documents
   - Supports PDF, DOCX, TXT, and Excel files
   ```bash
   curl -X POST "http://localhost:8000/upload" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_file.pdf"
   ```

2. **Query** (`POST /query`)
   - Search through processed documents
   - Returns relevant matches with similarity scores
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{"query": "your question here", "top_k": 5}'
   ```

3. **Health Check** (`GET /health`)
   - Check API status
   ```bash
   curl -X GET "http://localhost:8000/health"
   ```

## Technical Details

### Embedding Model

- Uses `all-MiniLM-L6-v2` from Sentence Transformers
- 384-dimensional embeddings
- Optimized for semantic similarity search

### Vector Store

- FAISS for efficient similarity search
- Persistent storage of indices and metadata
- L2 distance for similarity calculations

### File Processing

- Chunk size: 1000 characters (configurable)
- Intelligent sentence boundary detection
- Metadata preservation for source tracking

## Data Flow

1. **Document Upload**
   ```
   Upload → Process File → Generate Chunks → Create Embeddings → Store in FAISS
   ```

2. **Query Processing**
   ```
   Query → Generate Embedding → Search FAISS → Return Matches
   ```

## Configuration

Key configurations are stored in the respective modules:

- `vector_store.py`: Index and metadata paths
- `file_processor.py`: Chunk size and processing parameters
- `embeddings.py`: Model selection and parameters

## Error Handling

The API implements comprehensive error handling:
- File format validation
- Processing error catching
- Vector store operation safety checks
- Detailed error messages in responses

## Performance Considerations

- Uses async/await for non-blocking operations
- FAISS for efficient vector similarity search
- Batch processing for embeddings generation
- Efficient memory management for large files

## Security

- CORS middleware configured
- File type validation
- Temporary file cleanup
- Size limit on uploads (configurable)

## Development

### Adding New File Types

To add support for a new file type:

1. Add a new processor function in `file_processor.py`:
```python
def process_new_type(file_path: str, chunk_size: int) -> List[str]:
    # Implementation
    pass
```

2. Update the `process_file` function to include the new type:
```python
if file_extension == '.new_type':
    return process_new_type(file_path, chunk_size)
```

### Modifying Vector Store

To modify vector store behavior:

1. Update index configuration in `vector_store.py`
2. Modify metadata structure as needed
3. Update search parameters in `search_vector_store`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here] 