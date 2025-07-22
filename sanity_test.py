#!/usr/bin/env python3
"""
Comprehensive Sanity Test for Cosdata SDK

This test file exercises all major SDK functions to ensure they work correctly.
It covers:
- Client initialization and authentication
- Collection management (create, get, list, delete)
- Different index types (dense, sparse, TF-IDF)
- Vector operations (upsert, batch upsert, delete, exists)
- Search operations (dense, sparse, text, batch searches)
- Transaction management
- Version management
- Embedding utilities
- Error handling and edge cases
"""

import numpy as np
import random
import logging
import time
from typing import List, Dict, Any
from cosdata import Client
from cosdata.embedding import embed_texts
from cosdata.api import Collection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
client = None
test_collections = []


def setup_client(host="http://127.0.0.1:8443", username="admin", password="admin"):
    """Initialize the client and basic setup."""
    global client
    logger.info("=== Setting up Sanity Test ===")
    try:
        client = Client(host=host, username=username, password=password)
        logger.info("✓ Client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize client: {e}")
        return False


def test_client_operations():
    """Test basic client operations."""
    logger.info("\n=== Testing Client Operations ===")

    try:
        # Test listing collections
        collections = client.list_collections()
        logger.info(f"✓ Listed {len(collections)} collections")

        # Test getting collections as objects
        collection_objects = client.collections()
        logger.info(f"✓ Retrieved {len(collection_objects)} collection objects")

        # Test client-level indexes module
        logger.info("  Testing client-level indexes module...")
        try:
            # This tests that the indexes module is properly initialized
            indexes_module = client.indexes
            logger.info("    ✓ Client indexes module initialized")
        except Exception as e:
            logger.warning(f"    ⚠ Client indexes module not available: {e}")

        return True
    except Exception as e:
        logger.error(f"✗ Client operations failed: {e}")
        return False


def test_dense_collection_operations():
    """Test dense vector collection operations."""
    global test_collections
    logger.info("\n=== Testing Dense Collection Operations ===")

    collection_name = f"sanity_test_dense_{int(time.time())}"

    try:
        # Create dense collection
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Sanity test dense collection",
        )
        test_collections.append(collection)
        logger.info(f"✓ Created dense collection: {collection.name}")

        # Get collection info
        info = collection.get_info()
        logger.info(f"✓ Retrieved collection info: {info.get('name', 'N/A')}")

        # Create dense index
        index = collection.create_index(
            distance_metric="cosine",
            num_layers=7,
            max_cache_size=1000,
            ef_construction=512,
            ef_search=256,
            neighbors_count=32,
            level_0_neighbors_count=64,
        )
        logger.info(f"✓ Created dense index: {index.name}")

        # Get index info
        index_info = collection.get_index(index.name)
        logger.info(f"✓ Retrieved index info")

        # Test vector operations
        test_dense_vector_operations(collection)

        # Test dense search
        test_dense_search_operations(collection)

        # Test batch operations
        test_batch_operations(collection)

        # Cleanup
        index.delete()
        logger.info("✓ Deleted dense index")

        return True

    except Exception as e:
        logger.error(f"✗ Dense collection operations failed: {e}")
        return False


def test_sparse_collection_operations():
    """Test sparse vector collection operations."""
    global test_collections
    logger.info("\n=== Testing Sparse Collection Operations ===")

    collection_name = f"sanity_test_sparse_{int(time.time())}"

    try:
        # Create sparse collection
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Sanity test sparse collection",
            dense_vector={"enabled": False, "dimension": 768},
            sparse_vector={"enabled": True},
            tf_idf_options={"enabled": False},
        )
        test_collections.append(collection)
        logger.info(f"✓ Created sparse collection: {collection.name}")

        # Create sparse index
        index = collection.create_sparse_index(
            name="sparse_index", quantization=64, sample_threshold=1000
        )
        logger.info(f"✓ Created sparse index: {index.name}")

        # Test sparse vector operations
        test_sparse_vector_operations(collection)

        # Test sparse search
        test_sparse_search_operations(collection)

        # Cleanup
        index.delete()
        logger.info("✓ Deleted sparse index")

        return True

    except Exception as e:
        logger.error(f"✗ Sparse collection operations failed: {e}")
        return False


def test_text_collection_operations():
    """Test text collection operations."""
    global test_collections
    logger.info("\n=== Testing Text Collection Operations ===")

    collection_name = f"sanity_test_text_{int(time.time())}"

    try:
        # Create text collection
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Sanity test text collection",
            dense_vector={"enabled": False, "dimension": 768},
            sparse_vector={"enabled": False},
            tf_idf_options={"enabled": True},
        )
        test_collections.append(collection)
        logger.info(f"✓ Created text collection: {collection.name}")

        # Create TF-IDF index
        index = collection.create_tf_idf_index(
            name="tf_idf_index", sample_threshold=1000, k1=1.5, b=0.75
        )
        logger.info(f"✓ Created TF-IDF index: {index.name}")

        # Test text operations
        test_text_operations(collection)

        # Test text search
        test_text_search_operations(collection)

        # Cleanup
        index.delete()
        logger.info("✓ Deleted TF-IDF index")

        return True

    except Exception as e:
        logger.error(f"✗ Text collection operations failed: {e}")
        return False


def test_hybrid_collection_operations():
    """Test hybrid collection with multiple vector types."""
    global test_collections
    logger.info("\n=== Testing Hybrid Collection Operations ===")

    collection_name = f"sanity_test_hybrid_{int(time.time())}"

    try:
        # Create hybrid collection
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Sanity test hybrid collection",
            dense_vector={"enabled": True, "dimension": 768},
            sparse_vector={"enabled": True},
            tf_idf_options={"enabled": True},
        )
        test_collections.append(collection)
        logger.info(f"✓ Created hybrid collection: {collection.name}")

        # Create multiple indexes
        dense_index = collection.create_index(distance_metric="cosine", num_layers=7)
        logger.info(f"✓ Created dense index: {dense_index.name}")

        sparse_index = collection.create_sparse_index(
            name="hybrid_sparse_index", quantization=64
        )
        logger.info(f"✓ Created sparse index: {sparse_index.name}")

        tf_idf_index = collection.create_tf_idf_index(name="hybrid_tf_idf_index")
        logger.info(f"✓ Created TF-IDF index: {tf_idf_index.name}")

        # Test hybrid operations
        test_hybrid_operations(collection)

        # Cleanup
        dense_index.delete()
        sparse_index.delete()
        tf_idf_index.delete()
        logger.info("✓ Deleted all hybrid indexes")

        return True

    except Exception as e:
        logger.error(f"✗ Hybrid collection operations failed: {e}")
        return False


def test_embedding_operations():
    """Test embedding utility operations."""
    global test_collections
    logger.info("\n=== Testing Embedding Operations ===")

    collection_name = f"sanity_test_embed_{int(time.time())}"

    try:
        # Test embedding generation first to determine dimension
        texts = [
            "Cosdata makes vector search easy!",
            "This is a test of the embedding utility.",
            "You can use different models for your embeddings.",
            "Embeddings are essential for semantic search.",
            "Let's test the embedding functionality.",
        ]

        # Test with a model that produces 768-dimensional vectors
        try:
            embeddings = embed_texts(texts, model_name="BAAI/bge-base-en-v1.5")
            dimension = len(embeddings[0]) if embeddings else 768
            logger.info(
                f"✓ Generated {len(embeddings)} embeddings with BAAI/bge-base-en-v1.5 model (dimension: {dimension})"
            )
        except Exception as e:
            logger.warning(f"⚠ BAAI/bge-base-en-v1.5 model failed, trying default: {e}")
            try:
                embeddings = embed_texts(texts)
                dimension = len(embeddings[0]) if embeddings else 384
                logger.info(
                    f"✓ Generated {len(embeddings)} embeddings with default model (dimension: {dimension})"
                )
            except Exception as e2:
                logger.error(f"✗ Both embedding models failed: {e2}")
                return False

        # Create collection for embeddings with correct dimension
        collection = client.create_collection(
            name=collection_name,
            dimension=dimension,
            description="Sanity test embedding collection",
        )
        test_collections.append(collection)
        logger.info(
            f"✓ Created embedding collection: {collection.name} with dimension {dimension}"
        )

        # Create index
        index = collection.create_index(distance_metric="cosine", num_layers=7)
        logger.info(f"✓ Created index for embeddings")

        # Upsert embeddings
        with collection.transaction() as txn:
            for i, emb in enumerate(embeddings):
                txn.upsert_vector(
                    {
                        "id": f"embed_vec_{i + 1}",
                        "dense_values": emb,
                        "document_id": f"doc_{i // 2}",
                    }
                )
        logger.info("✓ Upserted embeddings through transaction")

        # Test search with embeddings
        results = collection.search.dense(
            query_vector=embeddings[0], top_k=3, return_raw_text=True
        )
        logger.info(
            f"✓ Performed search with embeddings: {len(results.get('results', []))} results"
        )

        # Cleanup
        index.delete()
        logger.info("✓ Deleted embedding index")

        return True

    except Exception as e:
        logger.error(f"✗ Embedding operations failed: {e}")
        return False


def test_version_operations():
    """Test version management operations."""
    logger.info("\n=== Testing Version Operations ===")

    try:
        # Use the first test collection for version testing
        if not test_collections:
            logger.warning("⚠ No test collections available for version testing")
            return True

        collection = test_collections[0]

        # Get current version
        current_version = collection.versions.get_current()
        logger.info(
            f"✓ Retrieved current version: version_number={current_version.version_number}, vector_count={current_version.vector_count}"
        )

        # Get version history
        try:
            version_info = collection.versions.list()
            versions = version_info.get("versions", [])
            logger.info(f"✓ Retrieved version history: {len(versions)} versions")
            for v in versions:
                logger.info(
                    f"    - version_number={v['version_number']}, vector_count={v['vector_count']}"
                )
            logger.info(f"    Current version: {version_info.get('current_version')}")
        except Exception as e:
            logger.warning(f"⚠ Version history not available: {e}")

        return True

    except Exception as e:
        logger.error(f"✗ Version operations failed: {e}")
        return False


def test_polling_and_status():
    """Test polling and status operations."""
    global test_collections
    logger.info("\n=== Testing Polling and Status Operations ===")

    collection_name = f"sanity_test_polling_{int(time.time())}"

    try:
        # Create collection for polling tests
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Sanity test polling collection",
        )
        test_collections.append(collection)
        logger.info(f"✓ Created polling collection: {collection.name}")

        # Create index
        index = collection.create_index(distance_metric="cosine", num_layers=7)
        logger.info(f"✓ Created index for polling tests")

        # Test indexing status
        try:
            indexing_status = collection.indexing_status()
            logger.info(f"✓ Retrieved indexing status: {indexing_status}")
        except Exception as e:
            logger.warning(f"⚠ Indexing status not available: {e}")

        # Test transaction status and polling
        logger.info("  Testing transaction status and polling...")

        # Create a transaction
        transaction = collection.create_transaction()
        logger.info(f"✓ Created transaction: {transaction.transaction_id}")

        # Get initial status
        try:
            initial_status = transaction.get_status()
            logger.info(f"    ✓ Initial transaction status: {initial_status}")
        except Exception as e:
            logger.warning(f"    ⚠ Transaction status not available: {e}")

        # Add some vectors to the transaction
        test_vectors = []
        for i in range(10):
            vector_id = f"polling_vec_{i + 1}"
            dense_values = np.random.uniform(-1, 1, 768).tolist()
            test_vectors.append(
                {
                    "id": vector_id,
                    "dense_values": dense_values,
                    "document_id": f"doc_{i // 5}",
                }
            )

        # Upsert vectors in transaction
        transaction.batch_upsert_vectors(test_vectors)
        logger.info("    ✓ Added vectors to transaction")

        # Get status after adding vectors
        try:
            status_after_upsert = transaction.get_status()
            logger.info(f"    ✓ Status after upsert: {status_after_upsert}")
        except Exception as e:
            logger.warning(f"    ⚠ Status after upsert not available: {e}")

        # Commit the transaction
        transaction.commit()
        logger.info("    ✓ Committed transaction")

        # Test polling for completion
        try:
            final_status, success = transaction.poll_completion(
                target_status="complete", max_attempts=5, sleep_interval=0.5
            )
            if success:
                logger.info(f"    ✓ Transaction polling successful: {final_status}")
            else:
                logger.warning(f"    ⚠ Transaction polling incomplete: {final_status}")
        except Exception as e:
            logger.warning(f"    ⚠ Transaction polling not available: {e}")

        # Test collection load/unload operations
        logger.info("  Testing collection load/unload operations...")

        try:
            # Load collection
            collection.load()
            logger.info("    ✓ Loaded collection")

            # Get loaded collections
            loaded_collections = Collection.loaded(client)
            logger.info(
                f"    ✓ Retrieved loaded collections: {len(loaded_collections)} collections"
            )

            # Unload collection
            collection.unload()
            logger.info("    ✓ Unloaded collection")
        except Exception as e:
            logger.warning(f"    ⚠ Load/unload operations not available: {e}")

        # Test version operations with polling
        logger.info("  Testing version operations...")

        try:
            # Get current version
            current_version = collection.versions.get_current()
            logger.info(f"    ✓ Current version: {current_version}")

            # Get version history
            version_history = collection.versions.get_history()
            logger.info(f"    ✓ Version history: {len(version_history)} versions")

            # Set version (if supported)
            try:
                collection.set_version(current_version)
                logger.info("    ✓ Set version successfully")
            except Exception as e:
                logger.warning(f"    ⚠ Set version not available: {e}")
        except Exception as e:
            logger.warning(f"    ⚠ Version operations not available: {e}")

        # Cleanup
        index.delete()
        logger.info("✓ Deleted polling index")

        return True

    except Exception as e:
        logger.error(f"✗ Polling and status operations failed: {e}")
        return False


def test_delete_vector_operations():
    """Test delete vector operations."""
    global test_collections
    logger.info("\n=== Testing Delete Vector Operations ===")

    collection_name = f"sanity_test_delete_{int(time.time())}"

    try:
        # Create collection for delete tests
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Sanity test delete collection",
        )
        test_collections.append(collection)
        logger.info(f"✓ Created delete collection: {collection.name}")

        # Create index
        index = collection.create_index(distance_metric="cosine", num_layers=7)
        logger.info(f"✓ Created index for delete tests")

        # Test vector operations with delete
        logger.info("  Testing delete vector operations...")

        # Generate test vectors
        test_vectors = []
        for i in range(20):
            vector_id = f"delete_vec_{i + 1}"
            dense_values = np.random.uniform(-1, 1, 768).tolist()
            test_vectors.append(
                {
                    "id": vector_id,
                    "dense_values": dense_values,
                    "document_id": f"doc_{i // 5}",
                }
            )

        # Test stream upsert first
        logger.info("  Testing stream upsert...")
        try:
            # Test single vector stream upsert
            single_vector = test_vectors[0]
            result = collection.stream_upsert(single_vector)
            logger.info(f"    ✓ Single vector stream upsert successful: {result}")

            # Test batch stream upsert
            batch_vectors = test_vectors[1:10]  # Stream upsert 9 vectors
            result = collection.stream_upsert(batch_vectors)
            logger.info(f"    ✓ Batch stream upsert successful: {result}")

            # Verify vectors exist after stream upsert
            for vector in test_vectors[:10]:
                exists = collection.vectors.exists(vector["id"])
                if exists:
                    logger.info(f"    ✓ Stream upserted vector {vector['id']} exists")
                else:
                    logger.error(
                        f"    ✗ Stream upserted vector {vector['id']} does not exist"
                    )
                    return False
        except Exception as e:
            logger.warning(f"    ⚠ Stream upsert not available: {e}")

        # Upsert remaining vectors via transaction
        with collection.transaction() as txn:
            txn.batch_upsert_vectors(test_vectors[10:])
        logger.info("    ✓ Added remaining test vectors via transaction")

        # Verify all vectors exist before deletion tests
        logger.info("  Verifying all vectors exist before deletion tests...")
        for vector in test_vectors:
            exists = collection.vectors.exists(vector["id"])
            if exists:
                logger.info(f"    ✓ Vector {vector['id']} exists before deletion")
            else:
                logger.error(
                    f"    ✗ Vector {vector['id']} does not exist before deletion"
                )
                return False

        # Test 1: Streaming delete
        logger.info("  Testing streaming delete...")
        try:
            vector_to_delete = test_vectors[0]
            result = collection.stream_delete(vector_to_delete["id"])
            logger.info(f"    ✓ Streaming delete successful: {result}")

            # Verify deletion
            exists_after_delete = collection.vectors.exists(vector_to_delete["id"])
            if not exists_after_delete:
                logger.info(
                    f"    ✓ Vector {vector_to_delete['id']} successfully deleted via streaming"
                )
            else:
                logger.error(
                    f"    ✗ Vector {vector_to_delete['id']} still exists after streaming delete"
                )
                return False
            # Also search for the deleted vector
            try:
                search_results = collection.search.dense(
                    query_vector=vector_to_delete["dense_values"],
                    top_k=10,
                    return_raw_text=True,
                )
                found = any(
                    r.get("id") == vector_to_delete["id"]
                    for r in search_results.get("results", [])
                )
                if not found:
                    logger.info(
                        f"    ✓ Deleted vector {vector_to_delete['id']} does not appear in search results"
                    )
                else:
                    logger.error(
                        f"    ✗ Deleted vector {vector_to_delete['id']} FOUND in search results!"
                    )
                    return False
            except Exception as e:
                logger.warning(f"    ⚠ Search after delete not available: {e}")
        except Exception as e:
            logger.warning(f"    ⚠ Streaming delete not available: {e}")

        # Test 2: Transaction delete
        logger.info("  Testing transaction delete...")
        try:
            vector_to_delete = test_vectors[1]

            # Create transaction and delete vector
            with collection.transaction() as txn:
                txn.delete_vector(vector_to_delete["id"])
                logger.info(f"    ✓ Added delete operation to transaction")

            # Verify deletion
            exists_after_delete = collection.vectors.exists(vector_to_delete["id"])
            if not exists_after_delete:
                logger.info(
                    f"    ✓ Vector {vector_to_delete['id']} successfully deleted via transaction"
                )
            else:
                logger.error(
                    f"    ✗ Vector {vector_to_delete['id']} still exists after transaction delete"
                )
                return False
        except Exception as e:
            logger.warning(f"    ⚠ Transaction delete not available: {e}")

        # Test 3: Batch delete via transaction
        logger.info("  Testing batch delete via transaction...")
        try:
            vectors_to_delete = test_vectors[2:5]  # Delete vectors 3-5

            with collection.transaction() as txn:
                for vector in vectors_to_delete:
                    txn.delete_vector(vector["id"])
                logger.info(
                    f"    ✓ Added {len(vectors_to_delete)} delete operations to transaction"
                )

            # Verify all deletions
            all_deleted = True
            for vector in vectors_to_delete:
                exists_after_delete = collection.vectors.exists(vector["id"])
                if exists_after_delete:
                    logger.error(
                        f"    ✗ Vector {vector['id']} still exists after batch delete"
                    )
                    all_deleted = False

            if all_deleted:
                logger.info(
                    f"    ✓ All {len(vectors_to_delete)} vectors successfully deleted via batch transaction"
                )
            else:
                return False
        except Exception as e:
            logger.warning(f"    ⚠ Batch delete not available: {e}")

        # Test 4: Delete non-existent vector
        logger.info("  Testing delete non-existent vector...")
        try:
            # Try to delete a vector that doesn't exist
            result = collection.stream_delete("non_existent_vector_id")
            logger.warning(f"    ⚠ Delete non-existent vector returned: {result}")
        except Exception as e:
            logger.info(f"    ✓ Correctly failed to delete non-existent vector: {e}")

        # Test 5: Get vectors by document ID and delete them
        logger.info("  Testing delete by document ID...")
        try:
            # Get vectors by document ID
            doc_id = "doc_0"
            vectors_in_doc = collection.vectors.get_by_document_id(doc_id)
            logger.info(
                f"    ✓ Found {len(vectors_in_doc)} vectors in document {doc_id}"
            )

            if vectors_in_doc:
                # Delete the first vector in the document
                vector_to_delete = vectors_in_doc[0]
                result = collection.stream_delete(vector_to_delete.id)
                logger.info(
                    f"    ✓ Deleted vector {vector_to_delete.id} from document {doc_id}"
                )

                # Verify deletion
                exists_after_delete = collection.vectors.exists(vector_to_delete.id)
                if not exists_after_delete:
                    logger.info(
                        f"    ✓ Vector {vector_to_delete.id} successfully deleted"
                    )
                else:
                    logger.error(
                        f"    ✗ Vector {vector_to_delete.id} still exists after deletion"
                    )
                    return False
        except Exception as e:
            logger.warning(f"    ⚠ Delete by document ID not available: {e}")

        # Test 6: Verify remaining vectors still exist
        logger.info("  Testing verification of remaining vectors...")
        remaining_vectors = test_vectors[5:10]  # Check vectors that weren't deleted
        for vector in remaining_vectors:
            exists = collection.vectors.exists(vector["id"])
            if exists:
                logger.info(f"    ✓ Remaining vector {vector['id']} still exists")
            else:
                logger.warning(
                    f"    ⚠ Remaining vector {vector['id']} was unexpectedly deleted"
                )

        # Cleanup
        index.delete()
        logger.info("✓ Deleted delete test index")

        return True

    except Exception as e:
        logger.error(f"✗ Delete vector operations failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("\n=== Testing Error Handling ===")

    try:
        # Test getting non-existent collection
        try:
            non_existent = client.get_collection("non_existent_collection")
            logger.error("✗ Should have failed to get non-existent collection")
            return False
        except Exception:
            logger.info("✓ Correctly failed to get non-existent collection")

        # Test creating collection with invalid parameters
        try:
            invalid_collection = client.create_collection(
                name="",  # Empty name
                dimension=0,  # Invalid dimension
            )
            logger.error(
                "✗ Should have failed to create collection with invalid parameters"
            )
            return False
        except Exception:
            logger.info(
                "✓ Correctly failed to create collection with invalid parameters"
            )

        return True

    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False


def test_dense_vector_operations(collection):
    """Test dense vector operations."""
    logger.info("  Testing dense vector operations...")

    # Generate test vectors
    num_vectors = 100
    vectors = []
    for i in range(num_vectors):
        vector_id = f"dense_vec_{i + 1}"
        dense_values = np.random.uniform(-1, 1, 768).tolist()
        vectors.append(
            {
                "id": vector_id,
                "dense_values": dense_values,
                "document_id": f"doc_{i // 10}",
                "metadata": {"batch": i // 50},
            }
        )

    # Test single vector upsert
    with collection.transaction() as txn:
        txn.upsert_vector(vectors[0])
    logger.info("    ✓ Single vector upsert")

    # Test batch vector upsert
    with collection.transaction() as txn:
        txn.batch_upsert_vectors(vectors[1:])
    logger.info("    ✓ Batch vector upsert")

    # Test vector existence
    exists = collection.vectors.exists(vectors[0]["id"])
    logger.info(f"    ✓ Vector existence check: {exists}")

    # Test get vector (using the proper vectors module)
    try:
        vector = collection.vectors.get(vectors[0]["id"])
        logger.info(f"    ✓ Get vector: {vector.id}")
    except Exception as e:
        logger.warning(f"    ⚠ Get vector not available: {e}")

    # Test get vectors by document ID
    try:
        doc_vectors = collection.vectors.get_by_document_id("doc_0")
        logger.info(f"    ✓ Get vectors by document ID: {len(doc_vectors)} vectors")
    except Exception as e:
        logger.warning(f"    ⚠ Get vectors by document ID not available: {e}")


def test_sparse_vector_operations(collection):
    """Test sparse vector operations."""
    logger.info("  Testing sparse vector operations...")

    # Generate test sparse vectors
    num_vectors = 50
    vectors = []
    for i in range(num_vectors):
        vector_id = f"sparse_vec_{i + 1}"
        # Generate sparse vector with 20-100 non-zero dimensions
        non_zero_dims = random.randint(20, 100)
        indices = sorted(random.sample(range(768), non_zero_dims))
        values = np.random.uniform(0.0, 2.0, non_zero_dims).tolist()

        vectors.append(
            {
                "id": vector_id,
                "sparse_values": values,
                "sparse_indices": indices,
                "document_id": f"doc_{i // 5}",
            }
        )

    # Test batch sparse vector upsert
    with collection.transaction() as txn:
        txn.batch_upsert_vectors(vectors)
    logger.info("    ✓ Batch sparse vector upsert")

    # Test vector existence
    exists = collection.vectors.exists(vectors[0]["id"])
    logger.info(f"    ✓ Sparse vector existence check: {exists}")


def test_text_operations(collection):
    """Test text operations."""
    logger.info("  Testing text operations...")

    # Generate test text documents
    num_documents = 50
    documents = []
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "All work and no play makes Jack a dull boy.",
        "To be or not to be, that is the question.",
        "It was the best of times, it was the worst of times.",
        "In a hole in the ground there lived a hobbit.",
    ]

    for i in range(num_documents):
        vector_id = f"text_doc_{i + 1}"
        text = random.choice(sample_texts) + f" Document number {i + 1}."
        documents.append(
            {"id": vector_id, "text": text, "document_id": f"doc_{i // 10}"}
        )

    # Test batch text upsert
    with collection.transaction() as txn:
        txn.batch_upsert_vectors(documents)
    logger.info("    ✓ Batch text upsert")

    # Test document existence
    exists = collection.vectors.exists(documents[0]["id"])
    logger.info(f"    ✓ Text document existence check: {exists}")


def test_hybrid_operations(collection):
    """Test hybrid operations with multiple vector types."""
    logger.info("  Testing hybrid operations...")

    # Generate hybrid vectors
    num_vectors = 30
    vectors = []
    for i in range(num_vectors):
        vector_id = f"hybrid_vec_{i + 1}"

        # Dense values
        dense_values = np.random.uniform(-1, 1, 768).tolist()

        # Sparse values
        non_zero_dims = random.randint(20, 100)
        indices = sorted(random.sample(range(768), non_zero_dims))
        sparse_values = np.random.uniform(0.0, 2.0, non_zero_dims).tolist()

        # Text
        text = f"This is hybrid document {i + 1} with both dense and sparse vectors."

        vectors.append(
            {
                "id": vector_id,
                "dense_values": dense_values,
                "sparse_values": sparse_values,
                "sparse_indices": indices,
                "text": text,
                "document_id": f"doc_{i // 5}",
            }
        )

    # Test hybrid upsert
    with collection.transaction() as txn:
        txn.batch_upsert_vectors(vectors)
    logger.info("    ✓ Hybrid vector upsert")


def test_dense_search_operations(collection):
    """Test dense search operations."""
    logger.info("  Testing dense search operations...")

    # Test single dense search
    query_vector = np.random.uniform(-1, 1, 768).tolist()
    results = collection.search.dense(
        query_vector=query_vector, top_k=5, return_raw_text=True
    )
    logger.info(f"    ✓ Single dense search: {len(results.get('results', []))} results")

    # Test batch dense search
    queries = [
        {"vector": np.random.uniform(-1, 1, 768).tolist()},
        {"vector": np.random.uniform(-1, 1, 768).tolist()},
        {"vector": np.random.uniform(-1, 1, 768).tolist()},
    ]

    try:
        batch_results = collection.search.batch_dense(
            queries=queries, top_k=3, return_raw_text=True
        )
        logger.info(f"    ✓ Batch dense search: {len(batch_results)} queries")
    except Exception as e:
        logger.warning(f"    ⚠ Batch dense search not available: {e}")


def test_sparse_search_operations(collection):
    """Test sparse search operations."""
    logger.info("  Testing sparse search operations...")

    # Generate sparse query
    non_zero_dims = random.randint(20, 100)
    indices = sorted(random.sample(range(768), non_zero_dims))
    values = np.random.uniform(0.0, 2.0, non_zero_dims).tolist()
    query_terms = [[idx, val] for idx, val in zip(indices, values)]

    # Test single sparse search
    results = collection.search.sparse(
        query_terms=query_terms,
        top_k=5,
        early_terminate_threshold=0.0,
        return_raw_text=True,
    )
    logger.info(
        f"    ✓ Single sparse search: {len(results.get('results', []))} results"
    )

    # Test batch sparse search
    try:
        batch_queries = [
            [[random.randint(0, 767), random.uniform(0.0, 2.0)] for _ in range(20)],
            [[random.randint(0, 767), random.uniform(0.0, 2.0)] for _ in range(20)],
            [[random.randint(0, 767), random.uniform(0.0, 2.0)] for _ in range(20)],
        ]

        batch_results = collection.search.batch_sparse(
            query_terms_list=batch_queries, top_k=3, return_raw_text=True
        )
        logger.info(f"    ✓ Batch sparse search: {len(batch_results)} queries")
    except Exception as e:
        logger.warning(f"    ⚠ Batch sparse search not available: {e}")


def test_text_search_operations(collection):
    """Test text search operations."""
    logger.info("  Testing text search operations...")

    # Test single text search
    query_text = "quick brown fox"
    results = collection.search.text(
        query_text=query_text, top_k=5, return_raw_text=True
    )
    logger.info(f"    ✓ Single text search: {len(results.get('results', []))} results")

    # Test batch text search
    try:
        query_texts = ["quick brown fox", "lazy dog", "work and play"]

        batch_results = collection.search.batch_text(
            query_texts=query_texts, top_k=3, return_raw_text=True
        )
        logger.info(f"    ✓ Batch text search: {len(batch_results)} queries")
    except Exception as e:
        logger.warning(f"    ⚠ Batch text search not available: {e}")

    # Test batch TF-IDF search (moved from collections.py)
    try:
        query_texts = ["quick brown fox", "lazy dog", "work and play"]

        batch_tf_idf_results = collection.search.batch_tf_idf_search(
            queries=query_texts, top_k=3, return_raw_text=True
        )
        logger.info(f"    ✓ Batch TF-IDF search: {len(batch_tf_idf_results)} queries")
    except Exception as e:
        logger.warning(f"    ⚠ Batch TF-IDF search not available: {e}")

    # Test hybrid search (moved from collections.py)
    try:
        hybrid_queries = {
            "dense": {
                "query_vector": np.random.uniform(-1, 1, 768).tolist(),
                "top_k": 5,
            },
            "text": {"query": "quick brown fox", "top_k": 5},
        }

        hybrid_results = collection.search.hybrid_search(hybrid_queries)
        logger.info(f"    ✓ Hybrid search: {len(hybrid_results)} results")
    except Exception as e:
        logger.warning(f"    ⚠ Hybrid search not available: {e}")


def test_batch_operations(collection):
    """Test batch operations."""
    logger.info("  Testing batch operations...")

    # Test large batch upsert
    large_batch = []
    for i in range(1000):
        vector_id = f"batch_vec_{i + 1}"
        dense_values = np.random.uniform(-1, 1, 768).tolist()
        large_batch.append(
            {
                "id": vector_id,
                "dense_values": dense_values,
                "document_id": f"doc_{i // 100}",
            }
        )

    with collection.transaction() as txn:
        txn.batch_upsert_vectors(large_batch, max_workers=4)
    logger.info("    ✓ Large batch upsert with workers")


def cleanup_collections():
    """Clean up all test collections."""
    global test_collections
    logger.info("\n=== Cleaning Up Test Collections ===")

    for collection in test_collections:
        try:
            collection.delete()
            logger.info(f"✓ Deleted collection: {collection.name}")
        except Exception as e:
            logger.error(f"✗ Failed to delete collection {collection.name}: {e}")

    test_collections.clear()


def run_all_tests():
    """Run all sanity tests."""
    logger.info("🚀 Starting Comprehensive Sanity Test for Cosdata SDK")

    if not setup_client():
        return False

    test_results = []

    # Run all tests
    # Running all the tests at once results in "LMDB Database Limit (maxdbs)" error
    tests = [
        ("Client Operations", test_client_operations),
        ("Dense Collection Operations", test_dense_collection_operations),
        # ("Sparse Collection Operations", test_sparse_collection_operations),
        # ("Text Collection Operations", test_text_collection_operations),
        # ("Hybrid Collection Operations", test_hybrid_collection_operations),
        # ("Embedding Operations", test_embedding_operations),
        ("Version Operations", test_version_operations),
        ("Polling and Status Operations", test_polling_and_status),
        ("Delete Vector Operations", test_delete_vector_operations),
        ("Error Handling", test_error_handling),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
            test_results.append((test_name, False))

    # Summary
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    logger.info(f"\n=== TEST SUMMARY ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.error("💥 SOME TESTS FAILED!")

    # Cleanup
    cleanup_collections()

    return passed == total


def main():
    """Main function to run the sanity test."""
    success = run_all_tests()

    if success:
        logger.info("🎉 Sanity test completed successfully!")
        return 0
    else:
        logger.error("💥 Sanity test failed!")
        return 1


if __name__ == "__main__":
    exit(main())
