# test_sample.py
import numpy as np
import random
from cosdata.client import Client

# Initialize the client
client = Client(
    host="http://127.0.0.1:8443"
)

def generate_random_vector_with_id(id: int, length: int) -> dict:
    """Generate a random vector with the specified ID and dimension."""
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values}

# Configuration
vector_db_name = "testdb_sdk_2"
dimension = 768
description = "Test Cosdata SDK"

# Create collection and index
collection = client.create_collection(
    name=vector_db_name,
    dimension=dimension,
    description=description
)
index = collection.create_index(
    distance_metric="cosine"
)

# Generate 1000 random vectors
batch_vectors = [
    generate_random_vector_with_id(i+1, dimension) 
    for i in range(1000)
]

print(f"Generated {len(batch_vectors)} vectors")

# Upsert all vectors in a single transaction (SDK will handle batching)
with index.transaction() as txn:
    txn.upsert(batch_vectors)
    print(f"Upserting complete - all vectors inserted in a single transaction")

# Select a random vector from the batch to query
query_vector = random.choice(batch_vectors)
print(f"Querying with vector ID: {query_vector['id']}")

# Query the index
results = index.query(
    vector=query_vector["values"],
    nn_count=5
)
print(f"Query results: {results}")

# Get collection info
collection_info = collection.get_info()
print(f"Collection info: {collection_info}")

# List all collections
print("All collections:")
for coll in client.collections():
    print(f" - {coll.name} (dimension: {coll.dimension})")
