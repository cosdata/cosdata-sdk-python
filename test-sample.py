import numpy as np
import random
from cosdata.client import Client

def generate_random_vector_with_id(id: int, length: int) -> dict:
    """
    Generate a random dense vector with an associated document.
    
    Args:
        id (int): The unique identifier for the vector.
        length (int): The dimension of the vector.
    
    Returns:
        dict: A dictionary with 'id', 'values' (the dense vector), and 'document' containing text.
    """
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values, "document": {"text": f"Random doc {id}"}}

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
vector_db_name = "testdb_sdk_2"
dimension = 768
description = "Test Cosdata SDK (Dense + Sparse)"

# -----------------------------------------------------
# Create (or retrieve) collection and index
# -----------------------------------------------------
client = Client(host="http://127.0.0.1:8443")
collection = client.create_collection(
    name=vector_db_name,
    dimension=dimension,
    description=description
)
index = collection.create_index(distance_metric="cosine")

# -----------------------------------------------------
# Generate 1000 random vectors
# -----------------------------------------------------
batch_vectors = [generate_random_vector_with_id(i + 1, dimension) for i in range(1000)]
print(f"Generated {len(batch_vectors)} vectors.")

# -----------------------------------------------------
# Upsert vectors in a single transaction
# -----------------------------------------------------
with index.transaction() as txn:
    txn.upsert(batch_vectors)
    print(" Upserting complete - all vectors inserted in a single transaction")

# -----------------------------------------------------
# Define multiple text queries for hybrid search
# -----------------------------------------------------
queries = [
    "AI search engine example",
    "machine learning for document search",
    "cosdata hybrid search functionality",
    "semantic vector search with neural networks",
    "sparse indexing using BM25 technique"
]

# -----------------------------------------------------
# Perform hybrid search for each query
# -----------------------------------------------------
for query in queries:
    print(f"\n[HYBRID] Query with text: '{query}'")
    try:
        hybrid_results = collection.hybrid_search(query=query, alpha=0.5, top_k=5)
        print("Hybrid Query Results:")
        for res in hybrid_results:
            print(f"ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")
    except Exception as e:
        print("Hybrid search failed:", e)

# -----------------------------------------------------
# Print collection information
# -----------------------------------------------------
collection_info = collection.get_info()
print("\nCollection info:")
print(collection_info)

# -----------------------------------------------------
# List all collections in the database
# -----------------------------------------------------
print("\nAll collections:")
for coll in client.collections():
    print(f" - {coll.name} (dimension: {coll.dimension})")
