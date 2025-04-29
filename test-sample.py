# test_sample.py
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

    logger.info("\n=== Dense Vector Collection Management ===")
    # Create a new dense collection
    collection = client.create_collection(
        name=collection_name,
        dimension=dimension,
        description=description
    )
    logger.info(f"Created dense collection: {collection.name}")

    # List all collections
    collections = client.collections()
    logger.info("\nAll collections:")
    for coll in collections:
        logger.info(f" - {coll.name}")

    logger.info("\n=== Dense Index Management ===")
    # Create a dense vector index
    index = collection.create_index(
        distance_metric="cosine",
        num_layers=7,
        max_cache_size=1000,
        ef_construction=512,
        ef_search=256,
        neighbors_count=32,
        level_0_neighbors_count=64
    )
    logger.info(f"Created dense index: {index.name}")

    # Get index information
    index_info = collection.get_index(index.name)
    logger.info(f"\nIndex information: {index_info}")

    logger.info("\n=== Dense Vector Operations ===")
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
    logger.info(f"Generated {len(dense_vectors)} test dense vectors")

    # Add dense vectors through a transaction
    logger.info("Starting transaction...")
    with collection.transaction() as txn:
        txn.batch_upsert_vectors(dense_vectors)
    logger.info("Added dense vectors through transaction")

    # Verify vector existence
    test_vector_id = dense_vectors[0]["id"]
    exists = collection.vectors.exists(test_vector_id)
    logger.info(f"\nVector {test_vector_id} exists: {exists}")

    logger.info("\n=== Dense Search Operations ===")
    # Perform dense vector search
    dense_query_vector = generate_random_vector(dimension)
    dense_results = collection.search.dense(
        query_vector=dense_query_vector,
        top_k=5,
        return_raw_text=True
    )
    logger.info(f"Dense search results: {dense_results}")

    logger.info("\n=== Version Management ===")
    # Get current version
    current_version = collection.versions.get_current()
    logger.info(f"Current version: {current_version}")

    # Cleanup
    logger.info("\n=== Cleanup ===")
    # Delete the index
    index.delete()
    logger.info("Deleted dense index")

    # Delete the collection
    collection.delete()
    logger.info("Deleted collection")

if __name__ == "__main__":
    main()
