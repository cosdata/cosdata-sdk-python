# search.py
import json
import requests
from typing import Dict, Any, List, Optional, Union


class Search:
    """
    Search module for performing vector searches.
    """

    def __init__(self, collection):
        """
        Initialize the search module.

        Args:
            collection: Collection instance
        """
        self.collection = collection

    def dense(
        self, query_vector: List[float], top_k: int = 10, return_raw_text: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar vectors using dense vector representation.

        Args:
            query_vector: Vector to search for similar vectors
            top_k: Maximum number of results to return
            return_raw_text: Whether to include raw text in the response

        Returns:
            Search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/dense"
        data = {
            "query_vector": query_vector,
            "top_k": top_k,
            "return_raw_text": return_raw_text,
        }

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to search dense vector: {response.text}")

        return response.json()

    def batch_dense(
        self,
        queries: List[Dict[str, List[float]]],
        top_k: int = 10,
        return_raw_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform batch search for similar vectors using dense vector representation.

        Args:
            queries: List of query objects, each containing a "vector" field with the query vector
            top_k: Maximum number of results to return per query
            return_raw_text: Whether to include raw text in the response

        Returns:
            List of search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/batch-dense"

        # Validate that each query has a "vector" field
        for i, query in enumerate(queries):
            if "vector" not in query:
                raise ValueError(f"Query at index {i} must contain a 'vector' field")

        data = {"queries": queries, "top_k": top_k, "return_raw_text": return_raw_text}

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to perform batch dense search: {response.text}")

        return response.json()

    def sparse(
        self,
        query_terms: List[Dict[str, Union[int, float]]],
        top_k: int = 10,
        early_terminate_threshold: float = 0.0,
        return_raw_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors using sparse vector representation.

        Args:
            query_terms: Array of sparse vector entries, each with an index and value
            top_k: Maximum number of results to return
            early_terminate_threshold: Threshold for early termination of search
            return_raw_text: Whether to include raw text in the response

        Returns:
            Search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/sparse"
        data = {
            "query_terms": query_terms,
            "top_k": top_k,
            "early_terminate_threshold": early_terminate_threshold,
            "return_raw_text": return_raw_text,
        }

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to search sparse vector: {response.text}")

        return response.json()

    def batch_sparse(
        self,
        query_terms_list: List[List[Dict[str, Union[int, float]]]],
        top_k: int = 10,
        early_terminate_threshold: float = 0.0,
        return_raw_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform batch search for similar vectors using sparse vector representation.

        Args:
            query_terms_list: List of sparse vector queries
            top_k: Maximum number of results to return per query
            early_terminate_threshold: Threshold for early termination of search
            return_raw_text: Whether to include raw text in the response

        Returns:
            List of search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/batch-sparse"
        data = {
            "query_terms_list": query_terms_list,
            "top_k": top_k,
            "early_terminate_threshold": early_terminate_threshold,
            "return_raw_text": return_raw_text,
        }

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to perform batch sparse search: {response.text}")

        return response.json()

    def text(
        self, query_text: str, top_k: int = 10, return_raw_text: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar vectors using text search.

        Args:
            query_text: Text to search for
            top_k: Maximum number of results to return
            return_raw_text: Whether to include raw text in the response

        Returns:
            Search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/tf-idf"
        data = {"query": query_text, "top_k": top_k, "return_raw_text": return_raw_text}

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to search text: {response.text}")

        return response.json()

    def batch_text(
        self, query_texts: List[str], top_k: int = 10, return_raw_text: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform batch search for similar vectors using text search.

        Args:
            query_texts: List of text queries
            top_k: Maximum number of results to return per query
            return_raw_text: Whether to include raw text in the response

        Returns:
            List of search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/batch-tf-idf"
        data = {
            "queries": query_texts,  # Changed from "query_texts" to "queries"
            "top_k": top_k,
            "return_raw_text": return_raw_text,
        }

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to perform batch text search: {response.text}")

        return response.json()

    def hybrid_search(self, queries):
        """
        Perform a hybrid search on this collection.
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/hybrid"
        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(queries),
            verify=self.collection.client.verify_ssl,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to perform hybrid search: {response.text}")
        return response.json()

    def batch_tf_idf_search(self, queries, top_k=10, return_raw_text=False):
        """
        Perform batch tf-idf search on this collection.
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/batch-tf-idf"
        data = {"queries": queries, "top_k": top_k, "return_raw_text": return_raw_text}
        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to perform batch tf-idf search: {response.text}")
        return response.json()

