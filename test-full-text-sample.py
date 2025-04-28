# test-full-text-sample.py
import random
import re
import unicodedata
import sys
from typing import Dict, List, Set
from cosdata import Client

def get_all_punctuation() -> Set[str]:
    """Get all Unicode punctuation characters."""
    return set(
        chr(i)
        for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith("P")
    )

def remove_non_alphanumeric(text: str) -> str:
    """Remove all non-alphanumeric characters from text."""
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        text = re.sub(r"[^\w]", " ", text.lower())
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()

def generate_random_text(num_words: int = 50) -> str:
    """Generate random text with the specified number of words."""
    # Sample words for text generation
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
    ]
    return " ".join(random.choices(words, k=num_words))

def main():
    # Initialize the client
    client = Client(
        host="http://127.0.0.1:8443",
        username="admin",
        password="test_key"
    )

    # Configuration
    collection_name = "test_text_collection"
    dimension = 768
    description = "Test collection for full-text search operations"

    print("\n=== Text Collection Management ===")
    # Create a new text collection
    collection = client.collections.create(
        name=collection_name,
        dimension=dimension,  # Required at root level
        description=description,
        dense_vector={
            "enabled": False,
            "dimension": dimension
        },
        sparse_vector={
            "enabled": False
        },
        tf_idf_options={
            "enabled": True
        }
    )
    print(f"Created text collection: {collection.name}")

    # List all collections
    collections = client.collections.list()
    print("\nAll collections:")
    for coll in collections:
        print(f" - {coll.name} (tf-idf: {coll.tf_idf_options.get('enabled')})")

    print("\n=== TF-IDF Index Management ===")
    # Create a TF-IDF index with BM25 parameters
    tf_idf_index = client.indexes.create_tf_idf(
        collection_name=collection_name,
        name="tf_idf_index",
        sample_threshold=1000,
        k1=1.5,  # BM25 k1 parameter
        b=0.75   # BM25 b parameter
    )
    print(f"Created TF-IDF index: {tf_idf_index.name}")

    # Get index information
    indexes = client.indexes.get(collection_name)
    print(f"\nTF-IDF index information: {indexes}")

    print("\n=== Text Vector Operations ===")
    # Generate some test text documents
    num_documents = 1000
    text_documents = []
    for i in range(num_documents):
        vector_id = f"doc_{i+1}"
        # Generate text with varying lengths
        text = generate_random_text(random.randint(20, 100))
        text_documents.append({
            "id": vector_id,
            "text": text,
            "document_id": f"doc_{i//10}"  # Group documents
        })
    print(f"Generated {len(text_documents)} test documents")

    # Add text documents through a transaction
    with client.transactions.transaction(collection_name) as txn:
        for doc in text_documents:
            txn.add_vector(
                vector_id=doc["id"],
                text=doc["text"],
                document_id=doc["document_id"]
            )
    print("Added text documents through transaction")

    # Verify document existence
    test_doc_id = text_documents[0]["id"]
    exists = client.vectors.exists(collection_name, test_doc_id)
    print(f"\nDocument {test_doc_id} exists: {exists}")

    print("\n=== Text Search Operations ===")
    # Perform text search with different queries
    test_queries = [
        "the quick brown fox jumps over the lazy dog",
        "people into year your good some could",
        "back after use two how our work first",
        "even new want because any these give",
        "day most us the be to of and"
    ]
    
    for query in test_queries:
        print(f"\nSearch query: {query}")
        text_results = client.search.text(
            collection_name=collection_name,
            query_text=query,
            top_k=5,
            return_raw_text=True
        )
        print(f"Text search results: {text_results}")

    print("\n=== Version Management ===")
    # Get current version
    current_version = client.versions.get_current(collection_name)
    print(f"Current version: {current_version}")

    # Cleanup
    print("\n=== Cleanup ===")
    # Delete the index
    client.indexes.delete(collection_name, "tf_idf")
    print("Deleted TF-IDF index")

    # Delete the collection
    client.collections.delete(collection_name)
    print("Deleted collection")

if __name__ == "__main__":
    main() 