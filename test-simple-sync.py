#!/usr/bin/env python3
"""
Test script for simple streaming transaction methods.
"""

import logging
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cosdata import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_sync_methods():
    """Test simple streaming transaction methods."""
    
    # Initialize client
    client = Client(
        host="http://127.0.0.1:8443",
        username="admin",
        password="asdfg",
        verify=False
    )
    
    collection_name = "test_simple_sync"
    
    try:
        # Create collection
        logger.info(f"Creating collection: {collection_name}")
        collection = client.create_collection(
            name=collection_name,
            dimension=5,  # Small dimension for testing
            description="Test collection for simple sync methods"
        )
        
        # Test 1: Single vector sync_upsert
        logger.info("Test 1: Single vector sync_upsert")
        vector1 = {
            "id": "vector-1",
            "document_id": "doc-123",
            "dense_values": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {
                "category": "technology",
                "tags": "ai,machine-learning",
                "score": 95
            },
            "text": "Sample text content"
        }
        
        result = collection.stream_upsert(vector1)
        logger.info(f"Single vector upsert result: {result}")
        # result = collection.stream_delete(vector1["id"])
        # logger.info(f"Single vector delete result: {result}")
        print(f"--------------------------------")
        
        # Test 2: Multiple vectors stream_upsert
        logger.info("Test 2: Multiple vectors stream_upsert")
        vectors = [
            {
                "id": "vector-2",
                "document_id": "doc-124",
                "dense_values": [0.2, 0.3, 0.4, 0.5, 0.6],
                "metadata": {
                    "category": "science",
                    "tags": "research,data",
                    "score": 89
                },
                "text": "Another sample text"
            },
            {
                "id": "vector-3",
                "document_id": "doc-125",
                "dense_values": [0.3, 0.4, 0.5, 0.6, 0.7],
                "metadata": {
                    "category": "education",
                    "tags": "learning,tutorial",
                    "score": 92
                },
                "text": "Third sample text"
            }
        ]
        
        result = collection.stream_upsert(vectors)
        logger.info(f"Multiple vectors upsert result: {result}")
        
        logger.info("All simple sync method tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during simple sync method tests: {e}")
        raise
    finally:
        # Clean up - delete the collection
        try:
            logger.info(f"Cleaning up - deleting collection: {collection_name}")
            collection.delete()
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")

if __name__ == "__main__":
    test_simple_sync_methods() 