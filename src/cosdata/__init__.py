"""
Cosdata Vector Database Python SDK

This package provides a Python client for interacting with the Cosdata Vector Database API.
The client uses an object-oriented, chainable interface for a more intuitive experience.

Example:
    from cosdata import Client

    # Initialize client
    client = Client(
        host="http://localhost:8443",
        username="admin",
        password="admin"
    )

    # Create a collection
    collection = client.create_collection(
        name="my_collection",
        dimension=768,
        description="My vector collection"
    )

    # Create an index
    index = collection.create_index(
        distance_metric="cosine",
        num_layers=7
    )

    # Add vectors using a transaction
    with index.transaction() as txn:
        txn.upsert({
            "id": "doc1",
            "values": [0.1, 0.2, 0.3, ...],
            "metadata": {
                "title": "Sample Document",
                "category": "example"
            }
        })

    # Search for similar vectors
    results = collection.search.dense(
        query_vector=[0.1, 0.2, 0.3, ...],
        top_k=5
    )
"""

from .api import Client

__version__ = "0.2.0"
__all__ = ["Client"] 