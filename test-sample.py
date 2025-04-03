import numpy as np
from cosdata.client import Client


def generate_random_vector_with_id(doc_id: int, length: int) -> dict:
    """
    Generate a random dense vector with an associated document.
    """
    values = np.random.uniform(-1, 1, length).tolist()
    return {
        "id": doc_id,
        "values": values,
        "document": {"text": f"Random document number {doc_id}"}
    }


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
vector_db_name = "testdb_sdk_2"
dimension = 768
description = "Test Cosdata SDK: Dense + Sparse + Hybrid independently."

# -----------------------------------------------------
# Create Client and Collection
# -----------------------------------------------------
client = Client(host="http://127.0.0.1:8443")
collection = client.create_collection(
    name=vector_db_name,
    dimension=dimension,
    description=description
)

# -----------------------------------------------------
# Create Indexes
# -----------------------------------------------------
print("\nCreating indexes...")
dense_index = collection.create_dense_index()
print("- Dense index created.")
sparse_index = collection.create_sparse_index()
print("- Sparse index created.")
# hybrid_index = collection.create_hybrid_index()
# print("- Hybrid index created.")

# -----------------------------------------------------
# Generate and Upsert Vectors
# -----------------------------------------------------
batch_vectors = [generate_random_vector_with_id(i + 1, dimension) for i in range(1000)]
print(f"\nGenerated {len(batch_vectors)} vectors.")

print("\nUpserting vectors into dense index...")
with dense_index.transaction() as txn:
    txn.upsert(batch_vectors)
print("- All vectors upserted in a single transaction.")


queries = [
    "AI search engine example",
    "machine learning for document search",
    "cosdata hybrid search functionality",
    "semantic vector search with neural networks",
    "sparse indexing using BM25 technique"
]


for query in queries:
    print(f"\n[QUERY] {query}")

    try:
        print("\nDense Search:")
        dense_results = collection.dense_search(top_k=5)
        for res in dense_results:
            print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")

        print("\nSparse Search:")
        sparse_results = collection.sparse_search(query=query, top_k=5)
        for res in sparse_results:
            print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")

        print("\nHybrid Search:")
        hybrid_results = collection.hybrid_search(query=query, alpha=0.5, top_k=5)
        for res in hybrid_results:
            print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")

    except Exception as e:
        print(f"Search failed for query '{query}':", str(e))


print("\nCollection Info:")
print(collection.get_info())


print("\nAll Collections:")
for coll in client.collections():
    print(f"- {coll.name} (dimension: {coll.dimension})")