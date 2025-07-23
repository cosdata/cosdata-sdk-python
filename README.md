# Cosdata Python SDK

A Python SDK for interacting with the Cosdata Vector Database.

## Installation

```bash
pip install cosdata-sdk
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
    # Single vector upsert (creates or updates)
    txn.upsert_vector(vectors[0])
    # Single vector create (only for new vectors)
    txn.create_vector(vectors[1])
    # Batch upsert for remaining vectors
    txn.batch_upsert_vectors(vectors[2:], max_workers=8, max_retries=3)

# Add vectors using streaming operations (immediate availability)
# Single vector upsert - returns immediately with result
result = collection.stream_upsert(vectors[0])
print(f"Stream upsert result: {result}")

# Multiple vectors upsert - returns immediately with result
result = collection.stream_upsert(vectors[1:])
print(f"Stream batch upsert result: {result}")

# Delete vectors using streaming operations
result = collection.stream_delete("vector-1")
print(f"Stream delete result: {result}")

# Add vectors using streaming operations (immediate availability)
# Single vector upsert - returns immediately with result
result = collection.stream_upsert(vectors[0])
print(f"Stream upsert result: {result}")

# Multiple vectors upsert - returns immediately with result
result = collection.stream_upsert(vectors[1:])
print(f"Stream batch upsert result: {result}")

# Delete vectors using streaming operations
result = collection.stream_delete("vector-1")
print(f"Stream delete result: {result}")

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
- `list_collections() -> List[Dict[str, Any]]`
  - Returns a list of collection information dictionaries.
- `indexes` - Access to client-level index management (see Indexes section below)

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
- `create_sparse_index(name: str) -> Index`
  - Creates a sparse index for the collection.
- `create_tf_idf_index(name: str, sample_threshold: int = 1000, k1: float = 1.2, b: float = 0.75) -> Index`
  - Creates a TF-IDF index for the collection.
- `get_index(name: str) -> Index`
  - Returns an `Index` object for the given name.
- `get_info() -> dict`
  - Returns collection metadata as above.
- `delete() -> None`
  - Deletes the collection.
- `load() -> None`
  - Loads the collection into memory.
- `unload() -> None`
  - Unloads the collection from memory.
- `create_transaction() -> Transaction`
  - Creates a new transaction for this collection.
- `transaction() -> Transaction` (context manager)
  - Creates a transaction with automatic commit/abort.
- `stream_upsert(vectors: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]`
  - Immediate vector upsert with immediate availability.
- `stream_delete(vector_id: str) -> Dict[str, Any]`
  - Immediate vector deletion.
- `neighbors(vector_id: str) -> Dict[str, Any]`
  - Fetch neighbors for a given vector ID.
- `set_version(version: str) -> Dict[str, Any]`
  - Set the current version of the collection.
- `indexing_status() -> Dict[str, Any]`
  - Get the indexing status of this collection.
- `loaded(client) -> List[Dict[str, Any]]` (class method)
  - Get a list of all loaded collections.

### Indexes

Index management is handled directly through the Collection object.

```python
# Create a dense index
dense_index = collection.create_index(
    distance_metric="cosine",
    num_layers=7
)

# Create a sparse index
sparse_index = collection.create_sparse_index(
    name="my_sparse_index",
    quantization=64
)

# Create a TF-IDF index
tf_idf_index = collection.create_tf_idf_index(
    name="my_tf_idf_index",
    sample_threshold=1000
)

# Get an existing index
index = collection.get_index("my_index")

# Delete an index
index.delete()
```

Methods:
- `create_index(distance_metric: str = "cosine", num_layers: int = 7, max_cache_size: int = 1000, ef_construction: int = 512, ef_search: int = 256, neighbors_count: int = 32, level_0_neighbors_count: int = 64) -> Index`
  - Creates a dense vector index for the collection.
- `create_sparse_index(name: str, quantization: int = 64, sample_threshold: int = 1000) -> Index`
  - Creates a sparse index for the collection.
- `create_tf_idf_index(name: str, sample_threshold: int = 1000, k1: float = 1.2, b: float = 0.75) -> Index`
  - Creates a TF-IDF index for the collection.
- `get_index(name: str) -> Index`
  - Get an existing index by name.

### Index

The Index class represents an index in a collection.

```python
index = collection.get_index("my_index")
```

Methods:
- `delete() -> None`
  - Deletes this index.

### Transaction

The Transaction class provides methods for vector operations with clear semantics.

```python
with collection.transaction() as txn:
    txn.upsert_vector(vector)  # Single vector (create or update)
    txn.batch_upsert_vectors(vectors, max_workers=8, max_retries=3)  # Multiple vectors, with parallelism and retries
```

Methods:
- `upsert_vector(vector: Dict[str, Any]) -> None`
  - **Creates or updates** an existing vector. Use this when you want to ensure the vector exists regardless of whether it already does.
- `delete_vector(vector_id: str) -> None`
  - Deletes a vector by ID in the transaction.
- `batch_upsert_vectors(vectors: List[Dict[str, Any]], max_workers: Optional[int] = None, max_retries: int = 3) -> None`
  - `vectors`: List of vector dictionaries to upsert
  - `max_workers`: Number of threads to use for parallel upserts (default: all available CPU threads)
  - `max_retries`: Number of times to retry a failed batch (default: 3)
- `commit() -> None`
  - Commits the transaction.
- `abort() -> None`
  - Aborts the transaction.
- `get_status() -> Dict[str, Any]`
  - Gets the status of the transaction.
- `poll_completion(target_status: str = 'complete', max_attempts: int = 10, sleep_interval: float = 1.0) -> Tuple[str, bool]`
  - Polls transaction status until target status is reached or max attempts exceeded.

### Transaction Status Polling

The Transaction class provides methods for monitoring transaction status and polling for completion.

```python
# Create a transaction
with collection.transaction() as txn:
    # Get current transaction status
    status = txn.get_status()
    print(f"Transaction status: {status}")
    
    # Upsert some vectors
    txn.upsert_vector(vector)
    
    # Poll for completion with custom parameters
    final_status, success = txn.poll_completion(
        target_status="complete",
        max_attempts=20,
        sleep_interval=2.0
    )
    
    if success:
        print(f"Transaction completed with status: {final_status}")
    else:
        print(f"Transaction may not have completed. Final status: {final_status}")
```

Methods:
- `get_status(collection_name: str = None, transaction_id: str = None) -> str`
  - Get the current status of this transaction (or another, if specified)
  - Returns transaction status as a string
  - Throws exceptions for API errors with descriptive messages
  - Parameters:
    - `collection_name`: Name of the collection (default: this transaction's collection)
    - `transaction_id`: ID of the transaction to check (default: this transaction's ID)
- `poll_completion(target_status: str = 'complete', max_attempts: int = 10, sleep_interval: float = 1.0, collection_name: str = None, transaction_id: str = None) -> tuple[str, bool]`
  - Poll transaction status until target status is reached or max attempts exceeded
  - Returns tuple of `(final_status, success_boolean)`
  - Configurable polling parameters for different use cases
  - Provides real-time progress feedback via console output
  - Parameters:
    - `target_status`: Target status to wait for (default: 'complete')
    - `max_attempts`: Maximum number of polling attempts (default: 10)
    - `sleep_interval`: Time to sleep between attempts in seconds (default: 1.0)
    - `collection_name`: Name of the collection (default: this transaction's collection)
    - `transaction_id`: Transaction ID to poll (default: this transaction's ID)

### Streaming Operations (Implicit Transactions)

The streaming operations provide immediate vector availability optimized for streaming scenarios. These methods use implicit transactions that prioritize data availability over batch processing efficiency.

**Design Philosophy:**
- **Optimized for streaming scenarios** where individual records must become immediately searchable
- **Serves real-time monitoring systems, live content feeds, and streaming analytics**
- **Prioritizes data availability over batch processing efficiency**
- **Automatic transaction management** - no client-managed transaction boundaries
- **System automatically handles batching and version allocation**
- **Abstracts transactional complexity while preserving append-only semantics**

```python
# Single vector stream upsert - immediately available for search
vector = {
    "id": "vector-1",
    "document_id": "doc-123",
    "dense_values": [0.1, 0.2, 0.3, 0.4, 0.5],
    "metadata": {"category": "technology"},
    "text": "Sample text content"
}
result = collection.stream_upsert(vector)
print(f"Vector immediately available: {result}")

# Multiple vectors stream upsert
vectors = [vector1, vector2, vector3]
result = collection.stream_upsert(vectors)
print(f"All vectors immediately available: {result}")

# Single vector stream delete
result = collection.stream_delete("vector-1")
print(f"Vector immediately deleted: {result}")
```

Methods:
- `stream_upsert(vectors: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]`
  - Upsert vectors with immediate availability
  - Returns response data immediately
  - Accepts single vector dict or list of vector dicts
- `stream_delete(vector_id: str) -> Dict[str, Any]`
  - Delete a vector with immediate effect
  - Returns response data immediately
  - Accepts single vector ID

### Transaction Status Polling

The Transaction class provides methods for monitoring transaction status and polling for completion.

```python
# Create a transaction
with collection.transaction() as txn:
    # Get current transaction status
    status = txn.get_status()
    print(f"Transaction status: {status}")
    
    # Upsert some vectors
    txn.upsert_vector(vector)
    
    # Poll for completion with custom parameters
    final_status, success = txn.poll_completion(
        target_status="complete",
        max_attempts=20,
        sleep_interval=2.0
    )
    
    if success:
        print(f"Transaction completed with status: {final_status}")
    else:
        print(f"Transaction may not have completed. Final status: {final_status}")
```

Methods:
- `get_status(collection_name: str = None, transaction_id: str = None) -> str`
  - Get the current status of this transaction (or another, if specified)
  - Returns transaction status as a string
  - Throws exceptions for API errors with descriptive messages
  - Parameters:
    - `collection_name`: Name of the collection (default: this transaction's collection)
    - `transaction_id`: ID of the transaction to check (default: this transaction's ID)
- `poll_completion(target_status: str = 'complete', max_attempts: int = 10, sleep_interval: float = 1.0, collection_name: str = None, transaction_id: str = None) -> tuple[str, bool]`
  - Poll transaction status until target status is reached or max attempts exceeded
  - Returns tuple of `(final_status, success_boolean)`
  - Configurable polling parameters for different use cases
  - Provides real-time progress feedback via console output
  - Parameters:
    - `target_status`: Target status to wait for (default: 'complete')
    - `max_attempts`: Maximum number of polling attempts (default: 10)
    - `sleep_interval`: Time to sleep between attempts in seconds (default: 1.0)
    - `collection_name`: Name of the collection (default: this transaction's collection)
    - `transaction_id`: Transaction ID to poll (default: this transaction's ID)

### Streaming Operations (Implicit Transactions)

The streaming operations provide immediate vector availability optimized for streaming scenarios. These methods use implicit transactions that prioritize data availability over batch processing efficiency.

**Design Philosophy:**
- **Optimized for streaming scenarios** where individual records must become immediately searchable
- **Serves real-time monitoring systems, live content feeds, and streaming analytics**
- **Prioritizes data availability over batch processing efficiency**
- **Automatic transaction management** - no client-managed transaction boundaries
- **System automatically handles batching and version allocation**
- **Abstracts transactional complexity while preserving append-only semantics**

```python
# Single vector stream upsert - immediately available for search
vector = {
    "id": "vector-1",
    "document_id": "doc-123",
    "dense_values": [0.1, 0.2, 0.3, 0.4, 0.5],
    "metadata": {"category": "technology"},
    "text": "Sample text content"
}
result = collection.stream_upsert(vector)
print(f"Vector immediately available: {result}")

# Multiple vectors stream upsert
vectors = [vector1, vector2, vector3]
result = collection.stream_upsert(vectors)
print(f"All vectors immediately available: {result}")

# Single vector stream delete
result = collection.stream_delete("vector-1")
print(f"Vector immediately deleted: {result}")
```

Methods:
- `stream_upsert(vectors: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]`
  - Upsert vectors with immediate availability
  - Returns response data immediately
  - Accepts single vector dict or list of vector dicts
- `stream_delete(vector_id: str) -> Dict[str, Any]`
  - Delete a vector with immediate effect
  - Returns response data immediately
  - Accepts single vector ID

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
- `batch_dense(queries: List[Dict[str, List[float]]], top_k: int = 5, return_raw_text: bool = False) -> List[dict]`
  - Batch dense vector search. Each query must contain a "vector" field.
- `sparse(query_terms: List[dict], top_k: int = 5, early_terminate_threshold: float = 0.0, return_raw_text: bool = False) -> dict`
  - Same structure as above.
- `batch_sparse(query_terms_list: List[List[dict]], top_k: int = 5, early_terminate_threshold: float = 0.0, return_raw_text: bool = False) -> List[dict]`
  - Batch sparse vector search.
- `text(query_text: str, top_k: int = 5, return_raw_text: bool = False) -> dict`
  - Same structure as above.
- `batch_text(query_texts: List[str], top_k: int = 5, return_raw_text: bool = False) -> List[dict]`
  - Batch text search.
- `hybrid_search(queries: dict) -> dict`
  - Hybrid search combining dense and sparse queries.
- `batch_tf_idf_search(queries: List[str], top_k: int = 10, return_raw_text: bool = False) -> List[dict]`
  - Batch TF-IDF search.

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
          "version_number": int,
          "vector_count": int
        },
        ...
      ],
      "current_version": int
    }
    ```
- `get_current() -> Version`
  - Returns a `Version` dataclass object with attributes:
    ```python
    version.version_number: int
    version.vector_count: int
    ```
- `get(version_number: int) -> Version`
  - Same as above.

# Usage Examples

## Basic Usage

```python
from cosdata import Client
client = Client(host="http://localhost:8443", username="admin", password="admin")
collection = client.get_collection("my_collection")
```

## Get Collection Indexing Status
Get the current indexing status of a collection, including progress and statistics. Useful for monitoring background indexing operations.
```python
status = collection.indexing_status()
print("Indexing status:", status)
```

## List Loaded Collections
Retrieve a list of all collections currently loaded in memory. This is helpful for understanding which collections are ready for fast access.
```python
loaded = Collection.loaded(client)
print("Loaded collections:", loaded)
```

## Create Sparse Index
Create a sparse index for your collection to enable efficient sparse vector search. You can specify the index name and optional parameters.
```python
result = collection.create_sparse_index("my_sparse_index")
print("Sparse index creation result:", result)
```

## Hybrid Search
Perform a hybrid search that combines dense and sparse vector queries. This is useful for advanced retrieval scenarios where you want to leverage both types of features.
```python
hybrid_query = {
    "dense_query": [0.1, 0.2, ...],
    "sparse_query": [{"index": 1, "value": 0.5}],
    "top_k": 5
}
results = collection.search.hybrid_search(hybrid_query)
print("Hybrid search results:", results)
```

## Batch TF-IDF Search
Run a batch of TF-IDF (text) searches in a single call. This is efficient for evaluating multiple queries at once.
```python
batch_queries = ["text query 1", "text query 2"]
results = collection.search.batch_tf_idf_search(batch_queries, top_k=3)
print("Batch TF-IDF results:", results)
```

## Fetch Vector Neighbors
Retrieve the nearest neighbors for a given vector ID in your collection. Useful for similarity search and recommendations.
```python
neighbors = collection.neighbors("vector_id")
print("Neighbors:", neighbors)
```

## Set Current Version
Set the current active version of a collection. This is important for versioned data management and switching between different dataset states.
```python
set_result = collection.set_version("version_id")
print("Set current version result:", set_result)
```

## Delete Vector via Streaming Endpoint
Delete a vector by its ID using the streaming endpoint. This is a fast way to remove vectors without managing explicit transactions.
```python
collection.stream_delete("vector_id")
print("Deleted vector via streaming endpoint")
```
