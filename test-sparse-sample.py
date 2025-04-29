# test-sparse-sample.py
import numpy as np
import random
from cosdata import Client
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    logger.info("\n=== Sparse Vector Collection Management ===")
    # Create a new sparse collection
    collection = client.create_collection(
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
    logger.info(f"Created sparse collection: {collection.name}")

    # List all collections
    collections = client.collections()
    logger.info("\nAll collections:")
    for coll in collections:
        logger.info(f" - {coll.name}")

    logger.info("\n=== Sparse Index Management ===")
    # Create a sparse vector index
    index = collection.create_sparse_index(
        name="sparse_index",
        quantization=64,
        sample_threshold=1000
    )
    logger.info(f"Created sparse index: {index.name}")

    # Get index information
    index_info = collection.get_index(index.name)
    logger.info(f"\nIndex information: {index_info}")

    logger.info("\n=== Sparse Vector Operations ===")
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
    logger.info(f"Generated {len(sparse_vectors)} test sparse vectors")

    # Add sparse vectors through a transaction
    logger.info("Starting transaction...")
    with collection.transaction() as txn:
        txn.batch_upsert_vectors(sparse_vectors)
    logger.info("Added sparse vectors through transaction")

    # Verify vector existence
    test_vector_id = sparse_vectors[0]["id"]
    exists = collection.vectors.exists(test_vector_id)
    logger.info(f"\nVector {test_vector_id} exists: {exists}")

    logger.info("\n=== Sparse Search Operations ===")
    # Perform sparse vector search
    sparse_data = generate_random_sparse_vector(dimension)
    query_terms = [[idx, val] for idx, val in zip(sparse_data["indices"], sparse_data["values"])]
    logger.info(f"Query vector: {query_terms[:5]}...")  # Print first 5 terms for debugging
    sparse_results = collection.search.sparse(
        query_terms=query_terms,
        top_k=5,
        early_terminate_threshold=0.0,
        return_raw_text=True
    )
    logger.info(f"Sparse search results: {sparse_results}")

    logger.info("\n=== Version Management ===")
    # Get current version
    current_version = collection.versions.get_current()
    logger.info(f"Current version: {current_version}")

    # Cleanup
    logger.info("\n=== Cleanup ===")
    # Delete the index
    index.delete()
    logger.info("Deleted sparse index")

    # Delete the collection
    collection.delete()
    logger.info("Deleted collection")

if __name__ == "__main__":
    main() 