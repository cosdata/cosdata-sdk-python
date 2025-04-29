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
    txn.batch_upsert_vectors(vectors[1:])

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
- `create_collection(name: str, dimension: int = 1024, description: Optional[str] = None, dense_vector: Optional[Dict[str, Any]] = None, sparse_vector: Optional[Dict[str, Any]] = None, tf_idf_options: Optional[Dict[str, Any]] = None) -> Collection`
- `collections() -> List[Collection]`
- `get_collection(name: str) -> Collection`

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
- `create_index(distance_metric: str = "cosine", num_layers: int = 7, max_cache_size: int = 1000, ef_construction: int = 512, ef_search: int = 256, neighbors_count: int = 32, level_0_neighbors_count: int = 64) -> Index`
- `create_sparse_index(name: str, quantization: int = 64, sample_threshold: int = 1000) -> Index`
- `create_tf_idf_index(name: str, sample_threshold: int = 1000, k1: float = 1.2, b: float = 0.75) -> Index`
- `get_index(name: str) -> Index`
- `get_info() -> Dict[str, Any]`
- `delete() -> None`
- `load() -> None`
- `unload() -> None`
- `transaction() -> Transaction` (context manager)

### Transaction

The Transaction class provides methods for vector operations.

```python
with collection.transaction() as txn:
    txn.upsert_vector(vector)  # Single vector
    txn.batch_upsert_vectors(vectors)  # Multiple vectors
```

Methods:
- `upsert_vector(vector: Dict[str, Any]) -> None`
- `batch_upsert_vectors(vectors: List[Dict[str, Any]]) -> None`
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
- `dense(query_vector: List[float], top_k: int = 5, return_raw_text: bool = False) -> Dict[str, Any]`
- `sparse(query_terms: List[List[float]], top_k: int = 5, early_terminate_threshold: float = 0.0, return_raw_text: bool = False) -> Dict[str, Any]`
- `text(query_text: str, top_k: int = 5, return_raw_text: bool = False) -> Dict[str, Any]`

### Vectors

The Vectors class provides methods for vector operations.

```python
vector = collection.vectors.get("vec_1")
exists = collection.vectors.exists("vec_1")
```

Methods:
- `get(vector_id: str) -> Dict[str, Any]`
- `get_by_document_id(document_id: str) -> List[Dict[str, Any]]`
- `exists(vector_id: str) -> bool`

### Versions

The Versions class provides methods for version management.

```python
current_version = collection.versions.get_current()
all_versions = collection.versions.list()
```

Methods:
- `list() -> List[Dict[str, Any]]`
- `get_current() -> Dict[str, Any]`
- `get(version_hash: str) -> Dict[str, Any]`

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