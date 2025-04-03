import json
import requests
import numpy as np
from typing import Dict, Any, List

from .index import Index
from .vector_utils import process_sentence, construct_sparse_vector


class Collection:
    def __init__(self, client, name: str, dimension: int):
        self.client = client
        self.name = name
        self.dimension = dimension

    # -----------------------------------------------------------------
    # Index Creation Methods
    # -----------------------------------------------------------------
    def create_dense_index(
        self,
        distance_metric: str = "cosine",
        num_layers: int = 7,
        max_cache_size: int = 1000,
        ef_construction: int = 512,
        ef_search: int = 256,
        neighbors_count: int = 32,
        level_0_neighbors_count: int = 64,
        sample_threshold: int = 100
    ) -> Index:
        data = {
            "name": self.name,
            "distance_metric_type": distance_metric,
            "quantization": {
                "type": "auto",
                "properties": {"sample_threshold": sample_threshold}
            },
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
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create dense index: {response.text}")
        return Index(self.client, self)

    def create_sparse_index(
        self,
        distance_metric: str = "cosine",
        quantization: int = 64,
        sample_threshold: int = 1000,
        early_terminate_threshold: float = 0.5
    ) -> Index:
        data = {
            "name": self.name,
            "distance_metric_type": distance_metric,
            "quantization": quantization,
            "sample_threshold": sample_threshold,
            "early_terminate_threshold": early_terminate_threshold
        }
        url = f"{self.client.base_url}/collections/{self.name}/indexes/sparse"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create sparse index: {response.text}")
        return Index(self.client, self)

    def get_info(self) -> Dict[str, Any]:
        url = f"{self.client.base_url}/collections/{self.name}"
        response = requests.get(
            url, headers=self.client._get_headers(), verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get collection info: {response.text}")
        return response.json()

    # -----------------------------------------------------------------
    # Search Methods
    # -----------------------------------------------------------------
    def dense_search(self, query_vector: List[float] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        if query_vector is None:
            query_vector = np.random.uniform(-1, 1, self.dimension).tolist()
        payload = {
            "vector_db_name": self.name,
            "vector": query_vector,
            "nn_count": top_k
        }
        response = requests.post(
            f"{self.client.base_url}/search",
            headers=self.client._get_headers(),
            data=json.dumps(payload),
            verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Dense search failed: {response.text}")
        return self._parse_response(response.json())

    def sparse_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        sparse_tokens = process_sentence(query, language="english")
        sparse_vector, doc_length = construct_sparse_vector(sparse_tokens)
        sparse_vector_serializable = [[pair[0], float(pair[1])] for pair in sparse_vector]
        placeholder_dense = np.random.uniform(-1, 1, self.dimension).tolist()
        payload = {
            "vector_db_name": self.name,
            "vector": placeholder_dense,
            "sparse_vector": {
                "indices": [pair[0] for pair in sparse_vector_serializable],
                "values": [pair[1] for pair in sparse_vector_serializable],
                "doc_length": int(doc_length)
            },
            "nn_count": top_k
        }
        response = requests.post(
            f"{self.client.base_url}/search",
            headers=self.client._get_headers(),
            data=json.dumps(payload),
            verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Sparse search failed: {response.text}")
        return self._parse_response(response.json())

    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10) -> List[Dict[str, Any]]:
        sparse_tokens = process_sentence(query, language="english")
        sparse_vector, doc_length = construct_sparse_vector(sparse_tokens)
        sparse_vector_serializable = [[pair[0], float(pair[1])] for pair in sparse_vector]
        dense_vector = np.random.uniform(-1, 1, self.dimension).tolist()

        payload = {
            "vector_db_name": self.name,
            "vector": dense_vector,
            "sparse_vector": {
                "indices": [pair[0] for pair in sparse_vector_serializable],
                "values": [pair[1] for pair in sparse_vector_serializable],
                "doc_length": int(doc_length)
            },
            "nn_count": top_k,
            "hybrid_alpha": alpha
        }

        response = requests.post(
            f"{self.client.base_url}/search",
            headers=self.client._get_headers(),
            data=json.dumps(payload),
            verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Hybrid search failed: {response.text}")
        return self._parse_response(response.json())

    def _parse_response(self, result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict) and "RespVectorKNN" in result:
            knn_list = result["RespVectorKNN"].get("knn", [])
            return [
                {
                    "id": item[0],
                    "score": item[1].get("CosineSimilarity", 0),
                    "document": item[1].get("document")
                }
                for item in knn_list
            ]
        elif isinstance(result, list):
            return result[0] if (len(result) > 0 and isinstance(result[0], list)) else result
        else:
            return result