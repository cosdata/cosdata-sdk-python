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

# Method 1: Step-by-step approach
collection = client.create_collection(
    name=vector_db_name,
    dimension=dimension,
    description=description
)

index = collection.create_index(
    distance_metric="cosine"
)

# Generate 100 random vectors
batch_vectors = [
    generate_random_vector_with_id(i+1, dimension) 
    for i in range(100)
]

# Method 2: Using context manager for transaction
with index.transaction() as txn:
    txn.upsert(batch_vectors)
    # Auto-commits on successful exit

# Method 3: Chainable methods approach
# client.collection(vector_db_name).index().create_transaction().upsert(batch_vectors).commit()

# Select a random vector from the batch to query
query_vector = random.choice(batch_vectors)
print("Querying with vector ID:", query_vector["id"])

# Query the index
results = index.query(
    vector=query_vector["values"],
    nn_count=5
)
print("Query results:", results)

# Get collection info
collection_info = collection.get_info()
print("Collection info:", collection_info)

# Method 4: List all collections using iterator
print("All collections:")
for coll in client.collections():
    print(f" - {coll.name} (dimension: {coll.dimension})")

