# test-sparse-sample.py
import numpy as np
import random
from cosdata import Client

def generate_random_sparse_vector(dimension: int, non_zero_dims: int = 100) -> dict:
    """Generate a random sparse vector of the specified dimension."""
    # Generate a random number of non-zero dimensions between 20 and 100
    actual_non_zero_dims = random.randint(20, non_zero_dims)
    
    # Generate unique indices
    indices = sorted(random.sample(range(dimension), actual_non_zero_dims))
    
    # Generate values between 0 and 2.0
    values = np.random.uniform(0.0, 2.0, actual_non_zero_dims).tolist()
    
    return {
        "indices": indices,
        "values": values
    }

def main():
    # Initialize the client
    client = Client(
        host="http://127.0.0.1:8443",
        username="admin",
        password="test_key"
    )

    # Configuration
    collection_name = "test_sparse_collection"
    dimension = 768
    description = "Test collection for sparse vector operations"

    print("\n=== Sparse Vector Collection Management ===")
    # Create a new sparse collection
    collection = client.collections.create(
        name=collection_name,
        dimension=dimension,
        description=description,
        dense_vector={
            "enabled": False,
            "dimension": dimension
        },
        sparse_vector={
            "enabled": True
        },
        tf_idf_options={
            "enabled": False
        }
    )
    print(f"Created sparse collection: {collection.name}")

    # List all collections
    collections = client.collections.list()
    print("\nAll collections:")
    for coll in collections:
        print(f" - {coll.name} (sparse: {coll.sparse_vector.get('enabled')})")

    print("\n=== Sparse Index Management ===")
    # Create a sparse vector index
    sparse_index = client.indexes.create_sparse(
        collection_name=collection_name,
        name="sparse_index",
        quantization=64,
        sample_threshold=1000
    )
    print(f"Created sparse index: {sparse_index.name}")

    # Get index information
    indexes = client.indexes.get(collection_name)
    print(f"\nSparse index information: {indexes}")

    print("\n=== Sparse Vector Operations ===")
    # Generate some test sparse vectors
    num_vectors = 1000
    sparse_vectors = []
    for i in range(num_vectors):
        vector_id = f"sparse_vec_{i+1}"
        sparse_data = generate_random_sparse_vector(dimension)
        sparse_vectors.append({
            "id": vector_id,
            "sparse_values": sparse_data["values"],
            "sparse_indices": sparse_data["indices"],
            "document_id": f"doc_{i//10}"  # Group vectors into documents
        })
    print(f"Generated {len(sparse_vectors)} test sparse vectors")

    # Add sparse vectors through a transaction
    with client.transactions.transaction(collection_name) as txn:
        for vector in sparse_vectors:
            txn.add_vector(
                vector_id=vector["id"],
                sparse_values=vector["sparse_values"],
                sparse_indices=vector["sparse_indices"],
                document_id=vector["document_id"]
            )
    print("Added sparse vectors through transaction")

    # Verify vector existence
    test_vector_id = sparse_vectors[0]["id"]
    exists = client.vectors.exists(collection_name, test_vector_id)
    print(f"\nVector {test_vector_id} exists: {exists}")

    print("\n=== Sparse Search Operations ===")
    # Perform sparse vector search
    sparse_data = generate_random_sparse_vector(dimension)
    query_terms = [[idx, val] for idx, val in zip(sparse_data["indices"], sparse_data["values"])]
    print(f"Query vector: {query_terms[:5]}...")  # Print first 5 terms for debugging
    sparse_results = client.search.sparse(
        collection_name=collection_name,
        query_terms=query_terms,
        top_k=5,
        early_terminate_threshold=0.0,
        return_raw_text=True
    )
    print(f"Sparse search results: {sparse_results}")

    print("\n=== Version Management ===")
    # Get current version
    current_version = client.versions.get_current(collection_name)
    print(f"Current version: {current_version}")

    # Cleanup
    print("\n=== Cleanup ===")
    # Delete the index
    client.indexes.delete(collection_name, "sparse")
    print("Deleted sparse index")

    # Delete the collection
    client.collections.delete(collection_name)
    print("Deleted collection")

if __name__ == "__main__":
    main() 