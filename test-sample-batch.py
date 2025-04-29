# test_sample_batch.py
import numpy as np
from cosdata import Client
import sys
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_random_vector(dimension: int) -> list:
    """Generate a random vector of the specified dimension."""
    return np.random.uniform(-1, 1, dimension).tolist()

def create_test_collection(client: Client, name: str, dimension: int) -> Any:
    """Create a test collection with the specified parameters."""
    logger.info(f"Creating collection: {name}")
    return client.create_collection(
        name=name,
        dimension=dimension,
        description="Test collection for batch operations"
    )

def create_test_index(collection: Any, name: str) -> Any:
    """Create a test index for the collection."""
    logger.info(f"Creating index: {name}")
    return collection.create_index(
        distance_metric="cosine",
        num_layers=7,
        max_cache_size=1000,
        ef_construction=512,
        ef_search=256,
        neighbors_count=32,
        level_0_neighbors_count=64
    )

def generate_test_vectors(num_vectors: int, dimension: int) -> List[Dict[str, Any]]:
    """Generate test vectors with the specified parameters."""
    logger.info(f"Generating {num_vectors} test vectors")
    vectors = []
    for i in range(num_vectors):
        if i % 1000 == 0:
            logger.info(f"Generated {i} vectors...")
        vector_id = f"dense_vec_{i+1}"
        dense_values = generate_random_vector(dimension)
        vectors.append({
            "id": vector_id,
            "dense_values": dense_values,
            "document_id": f"doc_{i//100}",  # Group vectors into documents of 100
            "metadata": {
                "batch_id": i // 1000  # Group vectors into batches of 1000
            }
        })
    return vectors

def perform_test_queries(collection: Any, num_queries: int, dimension: int) -> List[Dict[str, Any]]:
    """Perform test queries on the collection."""
    logger.info(f"Performing {num_queries} test queries")
    results = []
    for i in range(num_queries):
        logger.info(f"Performing query {i+1}...")
        query_vector = generate_random_vector(dimension)
        result = collection.search.dense(
            query_vector=query_vector,
            top_k=5,
            return_raw_text=True
        )
        results.append(result)
    return results

def main():
    """Main test function."""
    logger.info("Starting test_sample_batch.py...")
    
    try:
        # Initialize the client
        logger.info("Initializing client...")
        client = Client(
            host="http://127.0.0.1:8443",
            username="admin",
            password="test_key"
        )
        logger.info("Client initialized successfully")

        # Configuration
        collection_name = "test_batch_collection"
        dimension = 768
        num_vectors = 10000
        num_queries = 3

        # Create collection and index
        collection = create_test_collection(client, collection_name, dimension)
        index = create_test_index(collection, "dense_index")

        # Generate vectors
        vectors = generate_test_vectors(num_vectors, dimension)
        
        # Create and use transaction for upserting vectors
        logger.info("Starting transaction...")
        transaction = collection.create_transaction()
        logger.info("Upserting vectors...")
        transaction.upsert(vectors)
        transaction.commit()
        logger.info("Successfully upserted all vectors")

        # Perform test queries
        results = perform_test_queries(collection, num_queries, dimension)
        for i, result in enumerate(results):
            logger.info(f"Query {i+1} results: {result}")

        # Cleanup
        logger.info("Cleaning up...")
        index.delete()
        collection.delete()
        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 