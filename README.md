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
collection = client.create_collection(
    name="my_collection",
    dimension=768,                  # Vector dimension
    description="My vector collection"
)

# Create an index (all parameters are optional)
index = collection.create_index(
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
with index.transaction() as txn:
    txn.upsert(vectors)

# Search for similar vectors
results = index.query(
    vector=vectors[0]["values"],  # Use first vector as query
    nn_count=5                    # Number of nearest neighbors
)

# Fetch a specific vector
vector = index.fetch_vector(vector_id="1")

# Get collection information
collection_info = collection.get_info()
print(f"Collection info: {collection_info}")

# List all collections
print("Available collections:")
for coll in client.collections():
    print(f" - {coll.name} (dimension: {coll.dimension})")
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
- `create_collection(name: str, dimension: int = 1024, description: Optional[str] = None) -> Collection`
- `get_collection(collection_name: str) -> Collection`
- `list_collections() -> requests.Response`
- `collections() -> Iterator[Collection]`

### Collection

Represents a collection in the vector database.

```python
collection = client.get_collection("my_collection")
```

Methods:
- `create_index(distance_metric: str = "cosine", ...) -> Index`
- `index(distance_metric: str = "cosine") -> Index`
- `get_info() -> Dict[str, Any]`

### Index

Manages indexes and vector operations.

```python
index = collection.create_index()
```

Methods:
- `create_transaction() -> Transaction`
- `transaction() -> Iterator[Transaction]` (context manager)
- `query(vector: List[float], nn_count: int = 5) -> Dict[str, Any]`
- `fetch_vector(vector_id: Union[str, int]) -> Dict[str, Any]`

### Transaction

Manages batch operations on vectors.

```python
# Using context manager (recommended)
with index.transaction() as txn:
    txn.upsert(vectors)

# Manual transaction management
txn = index.create_transaction()
txn.upsert(vectors)
txn.commit()  # or txn.abort()
```

Methods:
- `upsert(vectors: List[Dict[str, Any]]) -> Self`
- `commit() -> Optional[Dict[str, Any]]`
- `abort() -> Optional[Dict[str, Any]]`

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.