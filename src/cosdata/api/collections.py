# collections.py
import json
import requests
from typing import Dict, Any, List, Optional
from .indexes import Index
from .search import Search
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

    def create_index(
        self,
        distance_metric: str = "cosine",
        num_layers: int = 7,
        max_cache_size: int = 1000,
        ef_construction: int = 512,
        ef_search: int = 256,
        neighbors_count: int = 32,
        level_0_neighbors_count: int = 64
    ) -> Index:
        """
        Create a new index for this collection.
        
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
            "quantization": {
                "type": "auto",
                "properties": {
                    "sample_threshold": 100
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
                    "level_0_neighbors_count": level_0_neighbors_count
                }
            }
        }
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create index: {response.text}")
        
        return Index(self, data["name"])

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
        
        Returns:
            Dictionary containing collection information
        """
        if self._info is None:
            url = f"{self.client.base_url}/collections/{self.name}"
            response = requests.get(
                url, 
                headers=self.client._get_headers(), 
                verify=self.client.verify_ssl
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
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 204:
            raise Exception(f"Failed to delete collection: {response.text}")

    def load(self) -> None:
        """
        Load this collection into memory.
        """
        url = f"{self.client.base_url}/collections/{self.name}/load"
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to load collection: {response.text}")

    def unload(self) -> None:
        """
        Unload this collection from memory.
        """
        url = f"{self.client.base_url}/collections/{self.name}/unload"
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to unload collection: {response.text}")

    def create_transaction(self) -> Transaction:
        """
        Create a new transaction for this collection.
        
        Returns:
            Transaction object
        """
        return Transaction(self) 