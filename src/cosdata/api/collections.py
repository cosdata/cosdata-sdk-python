# collections.py
import json
import requests
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
from .indexes import Index
from .search import Search
from .vectors import Vectors
from .versions import Versions
from .transactions import Transaction


class Collection:
    """
    Represents a collection in the vector database.
    """

    def __init__(self, client, name: str):
        """
        Initialize a collection.

        Args:
            client: Client instance
            name: Name of the collection
        """
        self.client = client
        self.name = name
        self._info = None
        self.search = Search(self)  # Initialize search module
        self.vectors = Vectors(self)  # Initialize vectors module
        self.versions = Versions(self)  # Initialize versions module

    @contextmanager
    def transaction(self):
        """
        Create a transaction with context management.

        This allows for automatic commit on success or abort on exception.

        Example:
            with collection.transaction() as txn:
                txn.upsert_vector(vector)  # For single vector
                txn.batch_upsert_vectors(vectors)  # For multiple vectors
                # Auto-commits on exit or aborts on exception

        Yields:
            Transaction object
        """
        txn = self.create_transaction()
        try:
            yield txn
            txn.commit()
        except Exception:
            txn.abort()
            raise

    def create_index(
        self,
        distance_metric: str = "cosine",
        num_layers: int = 7,
        max_cache_size: int = 1000,
        ef_construction: int = 512,
        ef_search: int = 256,
        neighbors_count: int = 32,
        level_0_neighbors_count: int = 64,
    ) -> Index:
           """
        Create a new dense index for this collection.
        
        Args:
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            num_layers: Number of layers in the HNSW graph
            max_cache_size: Maximum cache size
            ef_construction: ef parameter for index construction
            ef_search: ef parameter for search
            neighbors_count: Number of neighbors to connect to
            level_0_neighbors_count: Number of neighbors at level 0
            
        Returns:
            Index object
            """
        url = f"{self.client.base_url}/collections/{self.name}/indexes/dense"
        data = {
            "name": f"{self.name}_index",
            "distance_metric_type": distance_metric,
            "quantization": {"type": "auto", "properties": {"sample_threshold": 100}},
            "index": {
                "type": "hnsw",
                "properties": {
                    "num_layers": num_layers,
                    "max_cache_size": max_cache_size,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                    "neighbors_count": neighbors_count,
                    "level_0_neighbors_count": level_0_neighbors_count,
                },
            },
        }

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl,
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create index: {response.text}")

        return Index(self, data["name"], "dense")

    def create_sparse_index(
        self, name: str, quantization: int = 64, sample_threshold: int = 1000
    ) -> Index:
        """
        Create a new sparse index for this collection.
        
        Args:
            name: Name of the index
            quantization: Quantization bit value (16, 32, 64, 128, or 256)
            sample_threshold: Number of vectors to sample for calibrating the index
            
        Returns:
            Index object
        """
        url = f"{self.client.base_url}/collections/{self.name}/indexes/sparse"
        data = {
            "name": name,
            "quantization": quantization,
            "sample_threshold": sample_threshold,
        }

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl,
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create sparse index: {response.text}")

        return Index(self, name, "sparse")

    def create_tf_idf_index(
        self, name: str, sample_threshold: int = 1000, k1: float = 1.2, b: float = 0.75
    ) -> Index:
        """
        Create a new TF-IDF index for this collection.
        
        Args:
            name: Name of the index
            sample_threshold: Number of documents to sample for calibrating the index
            k1: BM25 k1 parameter that controls term frequency saturation
            b: BM25 b parameter that controls document length normalization
            
        Returns:
            Index object
        """
        url = f"{self.client.base_url}/collections/{self.name}/indexes/tf-idf"
        data = {"name": name, "sample_threshold": sample_threshold, "k1": k1, "b": b}

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl,
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create TF-IDF index: {response.text}")

        return Index(self, name, "tf_idf")

    def get_index(self, name: str) -> Index:
        """
        Get an existing index.
        
        Args:
            name: Name of the index
            
        Returns:
            Index object
        """
        return Index(self, name)

    def get_info(self) -> Dict[str, Any]:
        """
        Get collection information.
        """
        if self._info is None:
            url = f"{self.client.base_url}/collections/{self.name}"
            response = requests.get(
                url, headers=self.client._get_headers(), verify=self.client.verify_ssl
            )

            if response.status_code != 200:
                raise Exception(f"Failed to get collection info: {response.text}")

            self._info = response.json()

        return self._info

    def delete(self) -> None:
        """
        Delete this collection.
        """
        url = f"{self.client.base_url}/collections/{self.name}"
        response = requests.delete(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )

        if response.status_code != 204:
            raise Exception(f"Failed to delete collection: {response.text}")

    def load(self) -> None:
        """
        Load this collection into memory.
        """
        url = f"{self.client.base_url}/collections/{self.name}/load"
        response = requests.post(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )

        if response.status_code != 200:
            raise Exception(f"Failed to load collection: {response.text}")

    def unload(self) -> None:
        """
        Unload this collection from memory.
        """
        url = f"{self.client.base_url}/collections/{self.name}/unload"
        response = requests.post(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )

        if response.status_code != 200:
            raise Exception(f"Failed to unload collection: {response.text}")

    def create_transaction(self) -> Transaction:
        """
        Create a new transaction for this collection.
        """
        return Transaction(self)

    def stream_upsert(self, vectors: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Upsert vectors in this collection using streaming transaction.
        Returns immediately with the result.
        """
        if isinstance(vectors, dict):
            vectors = [vectors]

        url = f"{self.client.base_url}/collections/{self.name}/streaming/upsert"
        data = {"vectors": vectors}

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl,
        )

        if response.status_code not in [200, 201, 204]:
            raise Exception(f"Failed to streaming upsert vectors: {response.text}")

        return response.json() if response.content else {}

    def stream_delete(self, vector_id: str) -> Dict[str, Any]:
        """
        Delete a vector from this collection using streaming transaction.
        Returns immediately with the result.
        """
        url = f"{self.client.base_url}/collections/{self.name}/streaming/vectors/{vector_id}"
        response = requests.delete(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )

        if response.status_code not in [200, 201, 204]:
            raise Exception(f"Failed to streaming delete vector: {response.text}")

        return response.json() if response.content else {}

    def neighbors(self, vector_id: str):
        """
        Fetch neighbors for a given vector ID in this collection.
        """
        url = f"{self.client.base_url}/collections/{self.name}/vectors/{vector_id}/neighbors"
        response = requests.get(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Failed to fetch vector neighbors: {response.text}")
        return response.json()

    def set_version(self, version: str):
        """
        Set the current version of the collection.
        """
        url = f"{self.client.base_url}/collections/{self.name}/versions/current"
        data = {"version": version}
        response = requests.put(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to set current version: {response.text}")
        return response.json()

    def indexing_status(self):
        """
        Get the indexing status of this collection.
        """
        url = f"{self.client.base_url}/collections/{self.name}/indexing_status"
        response = requests.get(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get indexing status: {response.text}")
        return response.json()

    @classmethod
    def loaded(cls, client):
        """
        Get a list of all loaded collections.
        """
        url = f"{client.base_url}/collections/loaded"
        response = requests.get(
            url, headers=client._get_headers(), verify=client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get loaded collections: {response.text}")
        return response.json()
