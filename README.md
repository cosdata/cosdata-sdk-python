# Cosdata Python SDK

A Python SDK for interacting with the Cosdata Vector Database.

## Installation

```bash
pip install cosdata-client
```

## Quick Start

```python
from cosdata.client import Client  # Import the Client class

# Initialize the client (all parameters are optional)
client = Client(
    host="http://127.0.0.1:8443",  # Default host
    username="admin",               # Default username
    password="admin",               # Default password
    verify=False                    # SSL verification
)

# Create a collection
collection = client.collections.create(
    name="my_collection",
    dimension=768,                  # Vector dimension
    description="My vector collection"
)

# Create an index (all parameters are optional)
index = client.indexes.create(
    collection_name="my_collection",
    distance_metric="cosine",       # Default: cosine
    num_layers=10,                   # Default: 10
    max_cache_size=1000,            # Default: 1000
    ef_construction=128,            # Default: 128
    ef_search=64,                  # Default: 64
    neighbors_count=32,             # Default: 32
    level_0_neighbors_count=64      # Default: 64
)

# Generate some vectors (example with random data)
import numpy as np

def generate_random_vector(id: int, dimension: int) -> dict:
    values = np.random.uniform(-1, 1, dimension).tolist()
    return {
        "id": id,
        "values": values,
        "metadata": {  # Optional metadata
            "created_at": "2024-03-20",
            "category": "example"
        }
    }

# Generate and insert vectors
vectors = [generate_random_vector(i, 768) for i in range(100)]

# Add vectors using a transaction
with client.transactions.create() as txn:
    txn.upsert("my_collection", vectors)

# Search for similar vectors
results = client.search.query(
    collection_name="my_collection",
    vector=vectors[0]["values"],  # Use first vector as query
    nn_count=5                    # Number of nearest neighbors
)

# Fetch a specific vector
vector = client.vectors.fetch(
    collection_name="my_collection",
    vector_id="1"
)

# Get collection information
collection_info = client.collections.get("my_collection")
print(f"Collection info: {collection_info}")

# List all collections
print("Available collections:")
for coll in client.collections.list():
    print(f" - {coll['name']} (dimension: {coll['dimension']})")

# Version management
version = client.versions.create(
    collection_name="my_collection",
    description="New version with updated vectors"
)
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

The client provides access to the following modules:
- `collections`: Collection management
- `transactions`: Batch vector operations
- `search`: Vector similarity search
- `indexes`: Index management
- `vectors`: Vector operations
- `versions`: Version management

### Collections

Methods:
- `create(name: str, dimension: int = 1024, description: Optional[str] = None) -> Dict[str, Any]`
- `get(collection_name: str) -> Dict[str, Any]`
- `list() -> List[Dict[str, Any]]`
- `delete(collection_name: str) -> None`

### Transactions

Methods:
- `create() -> Transaction`
- `commit(transaction_id: str) -> Dict[str, Any]`
- `abort(transaction_id: str) -> None`

### Search

Methods:
- `query(collection_name: str, vector: List[float], nn_count: int = 5) -> Dict[str, Any]`
- `query_sparse(collection_name: str, vector: Dict[str, float], nn_count: int = 5) -> Dict[str, Any]`

### Indexes

Methods:
- `create(collection_name: str, distance_metric: str = "cosine", ...) -> Dict[str, Any]`
- `get(collection_name: str) -> Dict[str, Any]`
- `delete(collection_name: str) -> None`

### Vectors

Methods:
- `fetch(collection_name: str, vector_id: Union[str, int]) -> Dict[str, Any]`
- `delete(collection_name: str, vector_id: Union[str, int]) -> None`

### Versions

Methods:
- `create(collection_name: str, description: Optional[str] = None) -> Dict[str, Any]`
- `get(collection_name: str, version_id: str) -> Dict[str, Any]`
- `list(collection_name: str) -> List[Dict[str, Any]]`
- `delete(collection_name: str, version_id: str) -> None`

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