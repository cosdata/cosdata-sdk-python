import logging
from cosdata import Client
from cosdata.embedding import embed_texts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the client
    client = Client(
        host="http://127.0.0.1:8443",
        username="admin",
        password="test_key"
    )

    # Configuration
    collection_name = "test_embed_collection"
    dimension = 768  # thenlper/gte-base has 768 dimensions
    description = "Test collection for embedding utility demonstration"

    logger.info("=== Creating Collection ===")
    collection = client.create_collection(
        name=collection_name,
        dimension=dimension,
        description=description
    )
    logger.info(f"Created collection: {collection.name}")

    logger.info("=== Creating Dense Index ===")
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

    # Example texts to embed
    texts = [
        "Cosdata makes vector search easy!",
        "This is a test of the embedding utility.",
        "You can use different models for your embeddings.",
        "Let's try a non-default model for demonstration.",
        "Embeddings are essential for semantic search."
    ]

    # Use a non-default model (e.g., 'thenlper/gte-base')
    model_name = "thenlper/gte-base"
    logger.info(f"Generating embeddings using model: {model_name}")
    embeddings = embed_texts(texts, model_name=model_name)
    logger.info(f"Generated {len(embeddings)} embeddings.")

    # Upsert embeddings into the collection
    logger.info("Upserting embeddings into the collection...")
    with collection.transaction() as txn:
        for i, emb in enumerate(embeddings):
            txn.upsert_vector({
                "id": f"embed_vec_{i+1}",
                "dense_values": emb,
                "document_id": f"doc_{i//2}"
            })
    logger.info("Upserted embeddings.")

    # Perform a search using the first embedding
    logger.info("Performing dense search with the first embedding...")
    results = collection.search.dense(
        query_vector=embeddings[0],
        top_k=3,
        return_raw_text=True
    )
    logger.info(f"Search results: {results}")

    # Cleanup
    logger.info("Cleaning up: deleting collection.")
    collection.delete()
    logger.info("Deleted collection.")

if __name__ == "__main__":
    main()
