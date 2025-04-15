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
        response = requests.post(url, headers=self.client._get_headers(), data=json.dumps(data), verify=self.client.verify_ssl)

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create dense index: {response.text}")
        
        return Index(self.client, self)

    def create_splade_index(self, quantization: int = 64, sample_threshold: int = 10) -> Index:
        """
        Creates a SPLADE sparse vector index for the collection.
        """
        data = {
            "name": self.name,
            "quantization": quantization,
            "sample_threshold": sample_threshold,
        }
        url = f"{self.client.base_url}/collections/{self.name}/indexes/sparse"
        print(f"[DEBUG] Creating SPLADE index - URL: {url}, Data: {data}")
        response = requests.post(url, headers=self.client._get_headers(), data=json.dumps(data), verify=self.client.verify_ssl)

        print(f"[DEBUG] Create SPLADE Index - Status Code: {response.status_code}, Response Text: {response.text}")

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create splade index: {response.text}")

        return Index(self.client, self)

    def get_info(self) -> Dict[str, Any]:
        """
        Retrieves information about the collection.
        """
        url = f"{self.client.base_url}/collections/{self.name}"
        response = requests.get(url, headers=self.client._get_headers(), verify=self.client.verify_ssl)

        if response.status_code != 200:
            raise Exception(f"Failed to get collection info: {response.text}")
        
        return response.json()

    # === Search Methods ===

    def dense_search(self, query_vector: List[float] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs a dense vector search using the new dedicated endpoint.
        """
        if query_vector is None:
            query_vector = np.random.uniform(-1, 1, self.dimension).tolist()
        payload = {
            "query_vector": query_vector,  # Updated key to match server expectations
            "top_k": top_k,               # Use top_k as the key if the endpoint expects that key
        }
        # Updated endpoint for dense search
        url = f"{self.client.base_url}/collections/{self.name}/search/dense"
        response = requests.post(url, headers=self.client._get_headers(), data=json.dumps(payload), verify=self.client.verify_ssl)
        if response.status_code != 200:
            raise Exception(f"Dense search failed: {response.text}")
        return self._parse_response(response.json())


    def splade_sparse_search(self, query: str, top_k: int = 10, early_terminate_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Performs a SPLADE sparse vector search using the dedicated endpoint.
        """
        print(f"[DEBUG] Performing SPLADE Sparse Search\n[DEBUG] Query: {query}")

        sparse_tokens = process_sentence(query, language="english")
        print(f"[DEBUG] Processed Tokens: {sparse_tokens}")

        sparse_vector, _ = construct_sparse_vector(sparse_tokens)
        print(f"[DEBUG] Constructed Sparse Vector: {sparse_vector}")

        sparse_vector_serializable = [[int(token_id), float(score)] for token_id, score in sparse_vector]
        print(f"[DEBUG] Serialized Sparse Vector: {sparse_vector_serializable}")

        payload = {
            "index_type": "sparse",
            "query_terms": sparse_vector_serializable,  # Changed key from "values" to "query_terms"
            "top_k": top_k,
            "early_terminate_threshold": early_terminate_threshold
        }

        print(f"[DEBUG] SPLADE Sparse Search Payload: {json.dumps(payload, indent=2)}")
        url = f"{self.client.base_url}/collections/{self.name}/search/sparse"
        print(f"[DEBUG] SPLADE Sparse Search POSTing to: {url}")
        response = requests.post(url, headers=self.client._get_headers(), data=json.dumps(payload), verify=self.client.verify_ssl)

        print(f"[DEBUG] SPLADE Sparse Search Response Code: {response.status_code}\n[DEBUG] SPLADE Sparse Search Response Body: {response.text}")

        if response.status_code != 200:
            raise Exception(f"SPLADE sparse search failed: {response.text}")

        return self._parse_response(response.json())


    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10, query_vector: List[float] = None) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search combining dense and SPLADE sparse search using the dedicated endpoint.
        """
        sparse_tokens = process_sentence(query, language="english")
        sparse_vector, _ = construct_sparse_vector(sparse_tokens)
        sparse_vector_serializable = [[int(token_id), float(score)] for token_id, score in sparse_vector]

        if query_vector is None:
            query_vector = np.random.uniform(-1, 1, self.dimension).tolist()

        payload = {
            "query_vector": query_vector,          # Use 'query_vector' as required
            "query_terms": sparse_vector_serializable,  # Use 'query_terms' for the sparse part
            "top_k": top_k,
            "alpha": alpha
        }

        url = f"{self.client.base_url}/collections/{self.name}/search/hybrid"
        response = requests.post(url, headers=self.client._get_headers(), data=json.dumps(payload), verify=self.client.verify_ssl)

        if response.status_code != 200:
            raise Exception(f"Hybrid search failed: {response.text}")

        return self._parse_response(response.json())


    # === Response Parser ===

    def _parse_response(self, result: Any) -> List[Dict[str, Any]]:
        """
        Parses the JSON response from the Cosdata server.
        """
        if isinstance(result, dict):
            if "RespVectorKNN" in result:
                knn_list = result["RespVectorKNN"].get("knn", [])
                if knn_list and isinstance(knn_list[0], list):
                    return [{
                        "id": item[0],
                        "score": item[1].get("CosineSimilarity", 0),
                        "document": item[1].get("document")
                    } for item in knn_list]
                elif knn_list and isinstance(knn_list[0], dict):
                    return knn_list
                else:
                    return []
            elif "results" in result:
                return result["results"]
            else:
                return [result]
        elif isinstance(result, list):
            return result
        else:
            return []
