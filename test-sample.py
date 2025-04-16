import numpy as np
import requests
from cosdata.client import Client
from cosdata.transaction import Transaction
from cosdata.vector_utils import process_sentence, construct_sparse_vector

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

vector_db_name = "testdb_sdk_2"
dimension = 768
description = "Test Cosdata SDK: Dense + SPLADE Sparse + IDF operations."
cosdata_host = "http://127.0.0.1:8443"
num_dense_vectors = 20
num_sparse_vectors = 20

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------

def generate_random_vector_with_id(doc_id: int, length: int) -> dict:
    """Generate a random dense vector with an associated document."""
    values = np.random.uniform(-1, 1, length).tolist()
    return {
        "id": doc_id,
        "values": values,
        "document": {"text": f"Random dense document number {doc_id}"}
    }

def generate_sparse_vector_from_text(doc_id: int, text_content: str) -> dict:
    """Generate a sparse vector for SPLADE from text content."""
    sparse_tokens = process_sentence(text_content, language="english")
    sparse_vector, _ = construct_sparse_vector(sparse_tokens)
    indices = [int(token_id) for token_id, score in sparse_vector]
    values = [float(score) for token_id, score in sparse_vector]
    return {
        "id": doc_id,
        "indices": indices,
        "values": values
    }

def generate_idf_document_with_id(doc_id: int, text_content: str) -> dict:
    """Generate a document with text for IDF upsert."""
    return {"id": doc_id, "text": text_content}

def delete_idf_index(collection):
    """
    Deletes the TF-IDF index from the collection.
    
    This sends an HTTP DELETE request to the endpoint:
    /collections/{collection_name}/indexes/tf_idf
    """
    url = f"{collection.client.base_url}/collections/{collection.name}/indexes/tf_idf"
    response = requests.delete(url, headers=collection.client._get_headers(), verify=collection.client.verify_ssl)
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to delete TF-IDF index: {response.text}")
    print("- Existing TF-IDF index deleted.")

# -----------------------------------------------------
# Generate Sentences for Sparse Vectors
# -----------------------------------------------------

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly transforming many industries.",
    "Natural language processing enables computers to understand human language.",
    "Search engines use complex algorithms to find relevant information.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning has achieved remarkable success in image recognition.",
    "Data science involves analyzing large datasets to extract insights.",
    "Cloud computing provides scalable and on-demand computing resources.",
    "The internet has revolutionized communication and access to information.",
    "Software engineering is concerned with the design and development of software systems.",
    "Cybersecurity is crucial for protecting computer systems and networks.",
    "Database management systems are used to store and retrieve data efficiently.",
    "Mobile computing has become an integral part of modern life.",
    "Human-computer interaction focuses on the design of user-friendly interfaces.",
    "Robotics involves the design, construction, operation, and application of robots.",
    "Computer vision enables computers to interpret and understand visual information.",
    "The World Wide Web is a global collection of interconnected documents and resources.",
    "Algorithms are step-by-step procedures for solving a problem.",
    "Programming languages are used to instruct computers to perform tasks.",
    "Operating systems manage computer hardware and software resources.",
]

# -----------------------------------------------------
# Generate Documents for IDF Index
# -----------------------------------------------------

idf_documents = [
    "This document is about artificial intelligence and machine learning.",
    "Another document discussing search engine technology and information retrieval.",
    "A paper on natural language understanding and processing.",
    "The topic of this text is computer vision and image recognition.",
    "More content related to cloud computing and distributed systems.",
    "Software engineering principles and best practices.",
    "An overview of cybersecurity threats and defenses.",
    "Database management and data warehousing concepts.",
    "The impact of mobile computing on society.",
    "Designing effective human-computer interfaces.",
    "Applications of robotics in various industries.",
    "Advanced algorithms and data structures.",
    "Fundamentals of programming languages and paradigms.",
    "The architecture and functions of operating systems.",
    "The history and evolution of the internet and the World Wide Web.",
    "Data science methodologies and tools.",
    "Deep learning models and their applications.",
    "Ethical considerations in artificial intelligence.",
    "The future of computing and emerging technologies.",
    "A general overview of computer science topics."
]

# -----------------------------------------------------
# Main Execution
# -----------------------------------------------------

if __name__ == "__main__":
    # Create Client
    client = Client(host=cosdata_host)

    # Create Collection (or get if it exists)
    try:
        collection = client.create_collection(name=vector_db_name, dimension=dimension, description=description)
        print(f"\nCreated collection: {collection.name}")
    except Exception as e:
        print(f"\nError creating collection (it might already exist): {e}")
        collection = None
        for coll in client.collections():
            if coll.name == vector_db_name:
                collection = coll
                break
        if not collection:
            raise Exception(f"Collection '{vector_db_name}' not found.")

    if collection:
        # Create Indexes using the new API endpoints
        print("\nCreating indexes...")
        try:
            dense_index = collection.create_dense_index()      # Uses /collections/{id}/indexes/dense
            print("- Dense index created.")
            splade_index = collection.create_splade_index()      # Uses /collections/{id}/indexes/sparse
            print("- SPLADE sparse index created.")
            try:
                idf_index = collection.create_idf_index()        # Uses /collections/{id}/indexes/tf-idf
                print("- IDF index created.")
            except Exception as e:
                if "409" in str(e):
                    print("TF-IDF index already exists. Deleting it first...")
                    delete_idf_index(collection)
                    idf_index = collection.create_idf_index()    # Retry creating the index
                    print("- IDF index created after deletion.")
                else:
                    raise e
        except Exception as e:
            print(f"- Error creating indexes: {e}")

        # Upsert Dense Vectors using transaction for dense index
        dense_vectors = [generate_random_vector_with_id(i + 1, dimension) for i in range(num_dense_vectors)]
        print(f"\nGenerated {len(dense_vectors)} dense vectors.")
        print("\nUpserting vectors into dense index (using dense transactions)...")
        with dense_index.transaction() as txn:
            txn.upsert(dense_vectors)
        print("- All dense vectors upserted.")

        # # Upsert Sparse Vectors for SPLADE index using sparse transaction
        splade_vectors = [
            generate_sparse_vector_from_text(i + 1 + num_dense_vectors, sentences[i])
            for i in range(len(sentences))
        ]
        print(f"\nGenerated {len(splade_vectors)} sparse vectors for SPLADE index.")
        print("\nUpserting vectors into SPLADE sparse index (using sparse transactions)...")
        with Transaction(client, vector_db_name, "sparse") as txn:
            txn.upsert(splade_vectors)
        print("- All sparse vectors upserted.")

        # # Upsert Documents for IDF Index using a tf-idf transaction
        idf_docs = [
            generate_idf_document_with_id(i + 1 + num_dense_vectors + len(splade_vectors), text)
            for i, text in enumerate(idf_documents)
        ]
        print(f"\nGenerated {len(idf_docs)} documents for IDF index.")
        print("\nUpserting documents into IDF index (using tf-idf transactions)...")
        with Transaction(client, vector_db_name, "tf_idf") as txn:
            txn.upsert(idf_docs)
        print("- All IDF documents upserted.")

        # Run Searches using updated endpoint methods
        queries = [
            "artificial intelligence and machine learning",
            "search engine technology",
            "random document"  # Query that might also match dense vectors conceptually
        ]

        print("\nRunning Searches...")
        for query in queries:
            print(f"\n[QUERY] {query}")

            # Dense Search (new method name reflecting /search/dense)
            print("\nDense Search:")
            dense_results = collection.dense_search(top_k=5)
            for res in dense_results:
                print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")

            # SPLADE Sparse Search (updated method for /search/sparse)
            print("\nSPLADE Sparse Search:")
            sparse_results = collection.splade_sparse_search(query=query, top_k=5)
            print(f"[DEBUG] SPLADE Sparse Search Results: {sparse_results}")
            for res in sparse_results:
                print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")

            print("\nTF-IDF Search:")
            ires = collection.idf_search(query, top_k=5)
            for r in ires:
                print(f"  ID: {r['id']} | Score: {r['score']:.4f} | Doc: {r.get('document')}")    

            # Hybrid Search (still under /search/hybrid)
            print("\nHybrid Search for Dense+ Sparse:")
            hybrid_results_DS = collection.hybrid_search_Dense_sparse(query=query, alpha=0.5, top_k=5)
            for res in hybrid_results_DS:
                print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")

            print("\nHybrid Search for Dense+ IDF:")
            hybrid_results_DI = collection.hybrid_search_Dense_idf(query=query, alpha=0.5, top_k=5)
            for res in hybrid_results_DI:
                print(f"  ID: {res['id']} | Score: {res['score']:.4f} | Doc: {res.get('document')}")    

        # Collection Info (using the unified endpoint)
        print("\nCollection Info:")
        try:
            print(collection.get_info())
        except AttributeError:
            print("Collection info method not available in your client.")

        # List All Collections
        print("\nAll Collections:")
        for coll in client.collections():
            print(f"- {coll.name} (dimension: {coll.dimension})")
