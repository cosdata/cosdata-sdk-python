# test_sample.py
import numpy as np
import random
from cosdata import Client

def generate_random_vector(dimension: int) -> list:
    """Generate a random vector of the specified dimension."""
    return np.random.uniform(-1, 1, dimension).tolist()

def main():
    # Initialize the client
    client = Client(
        host="http://127.0.0.1:8443",
        username="admin",
        password="test_key"
    )

    # Configuration
    collection_name = "test_dense_collection"
    dimension = 768
    description = "Test collection for SDK demonstration"

    print("\n=== Dense Vector Collection Management ===")
    # Create a new dense collection
    collection = client.collections.create(
        name=collection_name,
        dimension=dimension,
        description=description,
        dense_vector={
            "enabled": True,
            "dimension": dimension
        },
        sparse_vector={
            "enabled": False
        },
        tf_idf_options={
            "enabled": False
        }
    )
    print(f"Created dense collection: {collection.name}")

    # List all collections
    collections = client.collections.list()
    print("\nAll collections:")
    for coll in collections:
        print(f" - {coll.name} (dense: {coll.dense_vector.get('enabled')})")

    print("\n=== Dense Index Management ===")
    # Create a dense vector index
    dense_index = client.indexes.create_dense(
        collection_name=collection_name,
        name="dense_index",
        distance_metric="cosine",
        quantization_type="auto",
        sample_threshold=100,
        num_layers=7,
        max_cache_size=1000,
        ef_construction=512,
        ef_search=256,
        neighbors_count=32,
        level_0_neighbors_count=64
    )
    print(f"Created dense index: {dense_index.name}")

    # Get index information
    indexes = client.indexes.get(collection_name)
    print(f"\nDense index information: {indexes}")

    print("\n=== Dense Vector Operations ===")
    # Generate some test dense vectors
    num_vectors = 1000
    dense_vectors = []
    for i in range(num_vectors):
        vector_id = f"dense_vec_{i+1}"
        dense_values = generate_random_vector(dimension)
        dense_vectors.append({
            "id": vector_id,
            "dense_values": dense_values,
            "document_id": f"doc_{i//10}"  # Group vectors into documents
        })
    print(f"Generated {len(dense_vectors)} test dense vectors")

    # Add dense vectors through a transaction
    with client.transactions.transaction(collection_name) as txn:
        for vector in dense_vectors:
            txn.add_vector(
                vector_id=vector["id"],
                dense_values=vector["dense_values"],
                document_id=vector["document_id"]
            )
    print("Added dense vectors through transaction")

    # Verify vector existence
    test_vector_id = dense_vectors[0]["id"]
    exists = client.vectors.exists(collection_name, test_vector_id)
    print(f"\nVector {test_vector_id} exists: {exists}")

    print("\n=== Dense Search Operations ===")
    # Perform dense vector search
    dense_query_vector = generate_random_vector(dimension)
    dense_results = client.search.dense(
        collection_name=collection_name,
        query_vector=dense_query_vector,
        top_k=5,
        return_raw_text=True
    )
    print(f"Dense search results: {dense_results}")

    print("\n=== Version Management ===")
    # Get current version
    current_version = client.versions.get_current(collection_name)
    print(f"Current version: {current_version}")

    # Cleanup
    print("\n=== Cleanup ===")
    # Delete the index
    client.indexes.delete(collection_name, "dense")
    print("Deleted dense index")

    # Delete the collection
    client.collections.delete(collection_name)
    print("Deleted collection")

if __name__ == "__main__":
    main()
