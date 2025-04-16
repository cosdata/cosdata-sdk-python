import json
import requests
import numpy as np
from typing import Dict, Any, List
from .index import Index
from .vector_utils import process_sentence, construct_sparse_vector


class Collection:
    """
    Represents a collection in the Cosdata vector database.
    """

    def __init__(self, client, name: str, dimension: int):
        """
        Initializes a Collection object.

        Args:
            client: The Cosdata client instance.
            name: The name of the collection.
            dimension: The dimensionality of the dense vectors in the collection.
        """
        self.client = client
        self.name = name
        self.dimension = dimension
        print(f"[DEBUG] Initialized Collection: name={self.name}, dimension={self.dimension}")

    # === Index Creation Methods ===

    def create_dense_index(self, distance_metric: str = "cosine", num_layers: int = 7,
                           max_cache_size: int = 1000, ef_construction: int = 512,
                           ef_search: int = 256, neighbors_count: int = 32,
                           level_0_neighbors_count: int = 64, sample_threshold: int = 10) -> Index:
        data = {
            "name": self.name,
            "distance_metric_type": distance_metric,
            "quantization": {"type": "auto", "properties": {"sample_threshold": sample_threshold}},
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
        url = f"{self.client.base_url}/collections/{self.name}/indexes/dense"
        response = requests.post(url, headers=self.client._get_headers(), json=data, verify=self.client.verify_ssl)
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create dense index: {response.text}")
        return Index(self.client, self)

    def create_splade_index(self, quantization: int = 64, sample_threshold: int = 10) -> Index:
        """
        Creates a SPLADE sparse vector index for the collection.
        """
        data = {"name": self.name, "quantization": quantization, "sample_threshold": sample_threshold}
        url = f"{self.client.base_url}/collections/{self.name}/indexes/sparse"
        print(f"[DEBUG] Creating SPLADE index - URL: {url}, Data: {data}")
        response = requests.post(url, headers=self.client._get_headers(), json=data, verify=self.client.verify_ssl)
        print(f"[DEBUG] Create SPLADE Index - Status Code: {response.status_code}, Response Text: {response.text}")
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create splade index: {response.text}")
        return Index(self.client, self)

    def create_idf_index(
        self,
        max_vectors: int = 1000,
        sample_threshold: int = 10,
        store_raw_text: bool = False,
        k1: float = 1.2,
        b: float = 0.75
    ) -> Index:
        """
        Creates a TF-IDF (IDF) index for the collection.
        """
        data = {
            "name": self.name,
            "max_vectors": max_vectors,
            "sample_threshold": sample_threshold,
            "store_raw_text": store_raw_text,
            "k1": k1,
            "b": b,
        }
        url = f"{self.client.base_url}/collections/{self.name}/indexes/tf-idf"
        print(f"[DEBUG] Creating TF-IDF index - URL: {url}, Data: {data}")
        response = requests.post(url, headers=self.client._get_headers(), json=data, verify=self.client.verify_ssl)
        print(f"[DEBUG] Create TF-IDF Index - Status Code: {response.status_code}, Response Text: {response.text}")
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create TF-IDF index: {response.text}")
        return Index(self.client, self)

    def get_info(self) -> Dict[str, Any]:
        """Retrieves information about the collection."""
        url = f"{self.client.base_url}/collections/{self.name}"
        response = requests.get(url, headers=self.client._get_headers(), verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"Failed to get collection info: {response.text}")
        return response.json()

    # === Search Methods ===

    def dense_search(self, query_vector: List[float] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs a dense vector search."""
        if query_vector is None:
            query_vector = np.random.uniform(-1, 1, self.dimension).tolist()
        payload = {"query_vector": query_vector, "top_k": top_k}
        url = f"{self.client.base_url}/collections/{self.name}/search/dense"
        response = requests.post(url, headers=self.client._get_headers(), json=payload, verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"Dense search failed: {response.text}")
        return self._parse_response(response.json())

    def splade_sparse_search(self, query: str, top_k: int = 10, early_terminate_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Performs a SPLADE sparse search."""
        sparse_tokens = process_sentence(query, language="english")
        sparse_vector, _ = construct_sparse_vector(sparse_tokens)
        terms = [[int(tid), float(score)] for tid, score in sparse_vector]
        payload = {"index_type": "sparse", "query_terms": terms, "top_k": top_k, "early_terminate_threshold": early_terminate_threshold}
        url = f"{self.client.base_url}/collections/{self.name}/search/sparse"
        response = requests.post(url, headers=self.client._get_headers(), json=payload, verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"SPLADE sparse search failed: {response.text}")
        return self._parse_response(response.json())

    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10, query_vector: List[float] = None) -> List[Dict[str, Any]]:
        """Performs a hybrid (dense + sparse) search."""
        sparse_tokens = process_sentence(query, language="english")
        sparse_vector, _ = construct_sparse_vector(sparse_tokens)
        terms = [[int(tid), float(score)] for tid, score in sparse_vector]
        if query_vector is None:
            query_vector = np.random.uniform(-1, 1, self.dimension).tolist()
        payload = {"query_vector": query_vector, "query_terms": terms, "top_k": top_k, "alpha": alpha}
        url = f"{self.client.base_url}/collections/{self.name}/search/hybrid"
        response = requests.post(url, headers=self.client._get_headers(), json=payload, verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"Hybrid search failed: {response.text}")
        return self._parse_response(response.json())

    def idf_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs a TF-IDF document search."""
        payload = {"query": query, "top_k": top_k}
        url = f"{self.client.base_url}/collections/{self.name}/search/tf-idf"
        response = requests.post(url, headers=self.client._get_headers(), json=payload, verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"TF-IDF search failed: {response.text}")
        return self._parse_response(response.json())

    def idf_batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """Performs a batch TF-IDF document search."""
        payload = {"queries": queries, "top_k": top_k}
        url = f"{self.client.base_url}/collections/{self.name}/search/tf-idf/batch"
        response = requests.post(url, headers=self.client._get_headers(), json=payload, verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"Batch TF-IDF search failed: {response.text}")
        return response.json().get("results", [])

    def _parse_response(self, result: Any) -> List[Dict[str, Any]]:
        """Parses the JSON response from the Cosdata server."""
        if isinstance(result, dict):
            if "RespVectorKNN" in result:
                knn = result["RespVectorKNN"].get("knn", [])
                if knn and isinstance(knn[0], list):
                    return [{"id": item[0], "score": item[1].get("CosineSimilarity", 0), "document": item[1].get("document")} for item in knn]
                elif knn and isinstance(knn[0], dict):
                    return knn
                return []
            return result.get("results", [result])
        if isinstance(result, list):
            return result
        return []
