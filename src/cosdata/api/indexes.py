# indexes.py
import json
import requests
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from .transactions import Transaction

@dataclass
class DenseIndex:
    """
    Represents a dense vector index configuration.
    """
    name: str
    distance_metric_type: str
    quantization: Dict[str, Any]
    index: Dict[str, Any]

@dataclass
class SparseIndex:
    """
    Represents a sparse vector index configuration.
    """
    name: str
    quantization: int
    sample_threshold: int

@dataclass
class TfIdfIndex:
    """
    Represents a TF-IDF index configuration.
    """
    name: str
    sample_threshold: int
    k1: float
    b: float

class Indexes:
    """
    Indexes module for managing vector indexes.
    """
    
    def __init__(self, client):
        """
        Initialize the indexes module.
        
        Args:
            client: Client instance
        """
        self.client = client
    
    def create_dense(
        self,
        collection_name: str,
        name: str,
        distance_metric: str = "cosine",
        quantization_type: str = "auto",
        sample_threshold: int = 100,
        num_layers: int = 7,
        max_cache_size: int = 1000,
        ef_construction: int = 512,
        ef_search: int = 256,
        neighbors_count: int = 32,
        level_0_neighbors_count: int = 64
    ) -> DenseIndex:
        """
        Create a dense vector index.
        
        Args:
            collection_name: Name of the collection
            name: Name of the index
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            quantization_type: Type of quantization (auto or scalar)
            sample_threshold: Number of vectors to sample for automatic quantization
            num_layers: Number of layers in the HNSW graph
            max_cache_size: Maximum cache size
            ef_construction: ef parameter for index construction
            ef_search: ef parameter for search
            neighbors_count: Number of neighbors to connect to
            level_0_neighbors_count: Number of neighbors at level 0
            
        Returns:
            DenseIndex object
        """
        url = f"{self.client.base_url}/collections/{collection_name}/indexes/dense"
        data = {
            "name": name,
            "distance_metric_type": distance_metric,
            "quantization": {
                "type": quantization_type,
                "properties": {
                    "sample_threshold": sample_threshold
                }
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
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create dense index: {response.text}")
        
        return DenseIndex(
            name=name,
            distance_metric_type=distance_metric,
            quantization=data["quantization"],
            index=data["index"]
        )
    
    def create_sparse(
        self,
        collection_name: str,
        name: str,
        quantization: int = 64,
        sample_threshold: int = 1000
    ) -> SparseIndex:
        """
        Create a sparse vector index.
        
        Args:
            collection_name: Name of the collection
            name: Name of the index
            quantization: Quantization bit value (16, 32, 64, 128, or 256)
            sample_threshold: Number of vectors to sample for calibrating the index
            
        Returns:
            SparseIndex object
        """
        url = f"{self.client.base_url}/collections/{collection_name}/indexes/sparse"
        data = {
            "name": name,
            "quantization": quantization,
            "sample_threshold": sample_threshold
        }
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create sparse index: {response.text}")
        
        return SparseIndex(
            name=name,
            quantization=quantization,
            sample_threshold=sample_threshold
        )
    
    def create_tf_idf(
        self,
        collection_name: str,
        name: str,
        sample_threshold: int = 1000,
        k1: float = 1.2,
        b: float = 0.75
    ) -> TfIdfIndex:
        """
        Create a TF-IDF index.
        
        Args:
            collection_name: Name of the collection
            name: Name of the index
            sample_threshold: Number of documents to sample for calibrating the index
            k1: BM25 k1 parameter that controls term frequency saturation
            b: BM25 b parameter that controls document length normalization
            
        Returns:
            TfIdfIndex object
        """
        url = f"{self.client.base_url}/collections/{collection_name}/indexes/tf-idf"
        data = {
            "name": name,
            "sample_threshold": sample_threshold,
            "k1": k1,
            "b": b
        }
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create TF-IDF index: {response.text}")
        
        return TfIdfIndex(
            name=name,
            sample_threshold=sample_threshold,
            k1=k1,
            b=b
        )
    
    def get(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about all indexes defined for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing index information
        """
        url = f"{self.client.base_url}/collections/{collection_name}/indexes"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get indexes: {response.text}")
            
        return response.json()
    
    def delete(self, collection_name: str, index_type: str) -> None:
        """
        Delete an index from a collection.
        
        Args:
            collection_name: Name of the collection
            index_type: Type of index to delete ("dense", "sparse", or "tf_idf")
        """
        url = f"{self.client.base_url}/collections/{collection_name}/indexes/{index_type}"
        response = requests.delete(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 204:
            raise Exception(f"Failed to delete index: {response.text}")

class Index:
    """
    Represents an index in a collection.
    """
    
    def __init__(self, collection, name: str):
        """
        Initialize an index.
        
        Args:
            collection: Collection instance
            name: Name of the index
        """
        self.collection = collection
        self.name = name

    def create_transaction(self) -> Transaction:
        """
        Create a new transaction.
        
        Returns:
            Transaction object
        """
        return Transaction(self)

    def transaction(self, callback) -> Any:
        """
        Execute operations in a transaction.
        
        Args:
            callback: Function to execute in the transaction
            
        Returns:
            Result of the callback function
        """
        txn = self.create_transaction()
        try:
            result = callback(txn)
            txn.commit()
            return result
        except Exception as e:
            txn.abort()
            raise e

    def query(
        self,
        vector: List[float],
        nn_count: int = 5,
        return_raw_text: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar vectors.
        
        Args:
            vector: Query vector
            nn_count: Number of nearest neighbors to return
            return_raw_text: Whether to include raw text in results
            
        Returns:
            Search results
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/dense"
        data = {
            "query_vector": vector,
            "top_k": nn_count,
            "return_raw_text": return_raw_text
        }
        
        response = requests.post(
            url, 
            headers=self.collection.client._get_headers(), 
            data=json.dumps(data), 
            verify=self.collection.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to query vectors: {response.text}")
            
        return response.json()

    def fetch_vector(self, vector_id: Union[str, int]) -> Dict[str, Any]:
        """
        Fetch a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to fetch
            
        Returns:
            Vector data
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/vectors/{vector_id}"
        response = requests.get(
            url, 
            headers=self.collection.client._get_headers(), 
            verify=self.collection.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch vector: {response.text}")
            
        return response.json()

    def delete(self) -> None:
        """
        Delete this index.
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/indexes/dense"
        response = requests.delete(
            url, 
            headers=self.collection.client._get_headers(), 
            verify=self.collection.client.verify_ssl
        )
        
        if response.status_code != 204:
            raise Exception(f"Failed to delete index: {response.text}") 