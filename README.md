# Cosdata Python SDK

A Python SDK for interacting with the Cosdata Vector Database.

## Installation

```bash
pip install cosdata-client
```

## Quick Start

```python
from cosdata import Client  # Import the Client class

# Initialize the client (all parameters are optional)
client = Client(
    host="http://127.0.0.1:8443",  # Default host
    username="admin",               # Default username
    password="admin",               # Default password
    verify=False                    # SSL verification
)

# Create a collection
collection = client.create_collection(
    name="my_collection",
    dimension=768,                  # Vector dimension
    description="My vector collection"
)

# Create an index (all parameters are optional)
index = collection.create_index(
    distance_metric="cosine",       # Default: cosine
    num_layers=10,                  # Default: 10
    max_cache_size=1000,            # Default: 1000
    ef_construction=128,            # Default: 128
    ef_search=64,                   # Default: 64
    neighbors_count=32,             # Default: 32
    level_0_neighbors_count=64      # Default: 64
)

# Generate some vectors (example with random data)
import numpy as np

def generate_random_vector(id: int, dimension: int) -> dict:
    values = np.random.uniform(-1, 1, dimension).tolist()
    return {
        "id": f"vec_{id}",
        "dense_values": values,
        "document_id": f"doc_{id//10}",  # Group vectors into documents
        "metadata": {  # Optional metadata
            "created_at": "2024-03-20",
            "category": "example"
        }
    }

# Generate and insert vectors
vectors = [generate_random_vector(i, 768) for i in range(100)]

# Add vectors using a transaction
with collection.transaction() as txn:
    # Single vector upsert
    txn.upsert_vector(vectors[0])
    # Batch upsert for remaining vectors
    txn.batch_upsert_vectors(vectors[1:], max_workers=8, max_retries=3)

# Search for similar vectors
results = collection.search.dense(
    query_vector=vectors[0]["dense_values"],  # Use first vector as query
    top_k=5,                                  # Number of nearest neighbors
    return_raw_text=True
)

# Fetch a specific vector
vector = collection.vectors.get("vec_1")

# Get collection information
collection_info = collection.get_info()
print(f"Collection info: {collection_info}")

# List all collections
print("Available collections:")
for coll in client.collections():
    print(f" - {coll.name}")

# Version management
current_version = collection.versions.get_current()
print(f"Current version: {current_version}")
```

## 🧩 Embedding Generation (Optional Convenience Feature)

Cosdata SDK provides a convenience utility for generating embeddings using [cosdata-fastembed](https://github.com/cosdata/cosdata-fastembed). This is optional—if you already have your own embeddings, you can use those directly. If you want to generate embeddings in Python, you can use the following utility:

```python
from cosdata.embedding import embed_texts

texts = [
    "Cosdata makes vector search easy!",
    "This is a test of the embedding utility."
]
embeddings = embed_texts(texts, model_name="thenlper/gte-base")  # Specify any supported model
```

- See the [cosdata-fastembed supported models list](https://github.com/cosdata/cosdata-fastembed#supported-models) for available model names and dimensions.
- The output is a list of lists (one embedding per input text), ready to upsert into your collection.
- If `cosdata-fastembed` is not installed, a helpful error will be raised.

## Methods

### embed_texts

- `embed_texts(texts: List[str], model_name: str = "BAAI/bge-small-en-v1.5") -> List[List[float]]`
  - Generates embeddings for a list of texts using cosdata-fastembed. Returns a list of embedding vectors (as plain Python lists). Raises ImportError if cosdata-fastembed is not installed.

  Example:
  ```python
  from cosdata.embedding import embed_texts
  embeddings = embed_texts(["hello world"], model_name="thenlper/gte-base")
  ```

## API Reference

### Client

The main client for interacting with the Vector Database API.

```python
client = Client(
    host="http://127.0.0.1:8443",  # Optional
    username="admin",               # Optional
    password="admin",               # Optional
    verify=False                    # Optional
)
```

Methods:
- `create_collection(...) -> Collection`
  - Returns a `Collection` object. Collection info can be accessed via `collection.get_info()`:
    ```python
    {
      "name": str,
      "description": str,
      "dense_vector": {"enabled": bool, "dimension": int},
      "sparse_vector": {"enabled": bool},
      "tf_idf_options": {"enabled": bool}
    }
    ```
- `collections() -> List[Collection]`
  - Returns a list of `Collection` objects.
- `get_collection(name: str) -> Collection`
  - Returns a `Collection` object for the given name.

### Collection

The Collection class provides access to all collection-specific operations.

```python
collection = client.create_collection(
    name="my_collection",
    dimension=768,
    description="My collection"
)
```

Methods:
- `create_index(...) -> Index`
  - Returns an `Index` object. Index info can be fetched (if implemented) as:
    ```python
    {
      "dense": {...},
      "sparse": {...},
      "tf-idf": {...}
    }
    ```
- `create_sparse_index(...) -> Index`
- `create_tf_idf_index(...) -> Index`
- `get_index(name: str) -> Index`
- `get_info() -> dict`
  - Returns collection metadata as above.
- `delete() -> None`
- `load() -> None`
- `unload() -> None`
- `transaction() -> Transaction` (context manager)

### Transaction

The Transaction class provides methods for vector operations.

```python
with collection.transaction() as txn:
    txn.upsert_vector(vector)  # Single vector
    txn.batch_upsert_vectors(vectors, max_workers=8, max_retries=3)  # Multiple vectors, with parallelism and retries
```

Methods:
- `upsert_vector(vector: Dict[str, Any]) -> None`
- `batch_upsert_vectors(vectors: List[Dict[str, Any]], max_workers: Optional[int] = None, max_retries: int = 3) -> None`
  - `vectors`: List of vector dictionaries to upsert
  - `max_workers`: Number of threads to use for parallel upserts (default: all available CPU threads)
  - `max_retries`: Number of times to retry a failed batch (default: 3)
- `commit() -> None`
- `abort() -> None`

### Search

The Search class provides methods for vector similarity search.

```python
results = collection.search.dense(
    query_vector=vector,
    top_k=5,
    return_raw_text=True
)
```

Methods:
- `dense(query_vector: List[float], top_k: int = 5, return_raw_text: bool = False) -> dict`
  - Returns:
    ```python
    {
      "results": [
        {
          "id": str,
          "document_id": str,
          "score": float,
          "text": str | None
        },
        ...
      ]
    }
    ```
- `sparse(query_terms: List[dict], top_k: int = 5, early_terminate_threshold: float = 0.0, return_raw_text: bool = False) -> dict`
  - Same structure as above.
- `text(query_text: str, top_k: int = 5, return_raw_text: bool = False) -> dict`
  - Same structure as above.

### Vectors

The Vectors class provides methods for vector operations.

```python
vector = collection.vectors.get("vec_1")
exists = collection.vectors.exists("vec_1")
```

Methods:
- `get(vector_id: str) -> Vector`
  - Returns a `Vector` dataclass object with attributes:
    ```python
    vector.id: str
    vector.document_id: Optional[str]
    vector.dense_values: Optional[List[float]]
    vector.sparse_indices: Optional[List[int]]
    vector.sparse_values: Optional[List[float]]
    vector.text: Optional[str]
    ```
- `get_by_document_id(document_id: str) -> List[Vector]`
  - Returns a list of `Vector` objects as above.
- `exists(vector_id: str) -> bool`
  - Returns `True` if the vector exists, else `False`.

### Versions

The Versions class provides methods for version management.

```python
current_version = collection.versions.get_current()
all_versions = collection.versions.list()
```

Methods:
- `list() -> dict`
  - Returns:
    ```python
    {
      "versions": [
        {
          "hash": str,
          "version_number": int,
          "timestamp": int,
          "vector_count": int
        },
        ...
      ],
      "current_hash": str
    }
    ```
- `get_current() -> Version`
  - Returns a `Version` dataclass object with attributes:
    ```python
    version.hash: str
    version.version_number: int
    version.timestamp: int
    version.vector_count: int
    version.created_at: datetime  # property for creation time
    ```
- `get(version_hash: str) -> Version`
  - Same as above.

## Best Practices

1. **Connection Management**
   - Reuse the client instance across your application
   - The client automatically handles authentication and token management

2. **Vector Operations**
   - Use transactions for batch operations
   - The context manager (`with` statement) automatically handles commit/abort
   - Maximum batch size is 200 vectors per transaction

3. **Error Handling**
   - All operations raise exceptions on failure
   - Use try/except blocks for error handling
   - Transactions automatically abort on exceptions when using the context manager

4. **Performance**
   - Adjust index parameters based on your use case
   - Use appropriate vector dimensions
   - Consider batch sizes for large operations

5. **Version Management**
   - Create versions before major changes
   - Use versions to track collection evolution
   - Clean up old versions when no longer needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.