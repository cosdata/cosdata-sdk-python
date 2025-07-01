#!/usr/bin/env python3
"""
Test script for transaction status polling functionality.

This script demonstrates how to use the transaction status polling methods
to monitor long-running transactions in the Cosdata SDK.
"""

import logging
import time
import random
from typing import List, Dict, Any

from src.cosdata import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_vector(dimension: int) -> List[float]:
    """Generate a random test vector."""
    return [random.uniform(-1, 1) for _ in range(dimension)]

def generate_test_vectors(num_vectors: int, dimension: int, prefix: str = "test_vec") -> List[Dict[str, Any]]:
    """Generate test vectors for upserting with unique IDs."""
    vectors = []
    timestamp = int(time.time())
    for i in range(num_vectors):
        vector_id = f"{prefix}_{timestamp}_{i+1}"
        dense_values = generate_test_vector(dimension)
        vectors.append({
            "id": vector_id,
            "dense_values": dense_values,
            "document_id": f"doc_{timestamp}_{i//10}",  # Group vectors into documents
            "metadata": {
                "category": f"category_{i % 5}",
                "timestamp": timestamp
            }
        })
    return vectors

def create_test_collection(client: Client, collection_name: str, dimension: int) -> Any:
    """Create a test collection."""
    try:
        # Check if collection already exists
        collections = client.list_collections()
        existing_collection = next((coll for coll in collections if coll["name"] == collection_name), None)
        
        if existing_collection:
            logger.info(f"Collection {collection_name} already exists, using existing collection")
            return client.get_collection(collection_name)
        
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            dimension=dimension,
            description=f"Test collection for transaction status polling (dimension: {dimension})"
        )
        logger.info(f"Created collection: {collection.name}")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise

def test_transaction_status_polling():
    """Test the transaction status polling functionality."""
    logger.info("=== Testing Transaction Status Polling ===")
    
    try:
        # Initialize the client
        logger.info("Initializing client...")
        client = Client(
            host="http://127.0.0.1:8443",
            username="admin",
            password="admin"
        )
        logger.info("Client initialized successfully")

        # Configuration
        collection_name = "test_transaction_status"
        dimension = 768
        num_vectors = 100  # Small batch for testing

        # Create collection
        collection = create_test_collection(client, collection_name, dimension)
        
        # Generate test vectors with unique IDs
        vectors = generate_test_vectors(num_vectors, dimension, "test_vec")
        logger.info(f"Generated {len(vectors)} test vectors")

        # Test 1: Create a transaction and get its status
        logger.info("\n--- Test 1: Getting Transaction Status ---")
        with collection.transaction() as txn:
            logger.info(f"Created transaction with ID: {txn.transaction_id}")
            
            # Get initial status
            status = txn.get_status()
            logger.info(f"Initial transaction status: {status}")
            
            # Upsert some vectors
            logger.info("Upserting vectors...")
            txn.batch_upsert_vectors(vectors[:10])  # Upsert first 10 vectors
            
            # Get status after upsert
            status = txn.get_status()
            logger.info(f"Transaction status after upsert: {status}")
            
            # Transaction will auto-commit here
        
        # Test 2: Poll for transaction completion
        logger.info("\n--- Test 2: Polling for Transaction Completion ---")
        
        # Create another transaction
        with collection.transaction() as txn:
            logger.info(f"Created transaction with ID: {txn.transaction_id}")
            
            # Upsert more vectors
            logger.info("Upserting remaining vectors...")
            txn.batch_upsert_vectors(vectors[10:])
            
            # Get the transaction ID before it commits
            transaction_id = txn.transaction_id
            
            # Note: In a real scenario, you might want to commit the transaction
            # and then poll for completion. For this test, we'll simulate
            # polling while the transaction is still active.
            
            logger.info("Starting to poll for transaction completion...")
            final_status, success = txn.poll_completion(
                target_status="complete",
                max_attempts=5,
                sleep_interval=1.0
            )
            
            logger.info(f"Polling completed - Final status: {final_status}, Success: {success}")
            
            # Transaction will auto-commit here

        # Test 3: Test with non-existent transaction ID
        logger.info("\n--- Test 3: Testing with Non-existent Transaction ID ---")
        try:
            # Create a transaction to get access to the get_status method
            with collection.transaction() as txn:
                status = txn.get_status(transaction_id="non_existent_txn_123")
                logger.info(f"Status: {status}")
        except Exception as e:
            logger.info(f"Expected error for non-existent transaction: {e}")

        # Test 4: Test polling with custom parameters
        logger.info("\n--- Test 4: Testing Polling with Custom Parameters ---")
        
        # Generate new vectors with unique IDs for this test
        new_vectors = generate_test_vectors(5, dimension, "custom_test_vec")
        
        # Create a transaction and immediately commit it
        with collection.transaction() as txn:
            transaction_id = txn.transaction_id
            logger.info(f"Created transaction with ID: {transaction_id}")
            txn.upsert_vector(new_vectors[0])  # Upsert single vector
            # Transaction will auto-commit
        
        # Now poll for completion with different parameters
        logger.info("Polling with custom parameters (max_attempts=3, sleep_interval=0.5)...")
        
        # Create another transaction to access the poll_completion method
        with collection.transaction() as txn:
            final_status, success = txn.poll_completion(
                target_status="complete",
                max_attempts=3,
                sleep_interval=0.5,
                transaction_id=transaction_id  # Poll the previous transaction
            )
        
        logger.info(f"Custom polling completed - Final status: {final_status}, Success: {success}")

        # Cleanup
        logger.info("\n--- Cleanup ---")
        logger.info("Deleting test collection...")
        collection.delete()
        logger.info("Test collection deleted successfully")

        logger.info("\n=== All Transaction Status Polling Tests Completed Successfully ===")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

def main():
    """Main function to run the transaction status polling tests."""
    logger.info("Starting transaction status polling tests...")
    
    try:
        test_transaction_status_polling()
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise

if __name__ == "__main__":
    main() 