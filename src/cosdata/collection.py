# collection.py
import json
import requests
from typing import Dict, Any, Optional, List, Iterator
from .index import Index

class Collection:
    """
    Class for managing a collection in the vector database.
    """
    
    def __init__(self, client, name: str, dimension: int):
        """
        Initialize a Collection object.
        
        Args:
            client: VectorDBClient instance
            name: Name of the collection
            dimension: Dimensionality of vectors in this collection
        """
        self.client = client
        self.name = name
        self.dimension = dimension
    
    def index(self, distance_metric: str = "cosine") -> Index:
        """
        Get or create an index for this collection.
        
        Args:
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            
        Returns:
            Index object
        """
        # This is a simplified version - a real implementation might check if index exists
        return self.create_index(distance_metric=distance_metric)
    
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
        Create an index for this collection.
        
        Args:
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            num_layers: Number of layers in the HNSW graph
            max_cache_size: Maximum cache size
            ef_construction: ef parameter for index construction
            ef_search: ef parameter for search
            neighbors_count: Number of neighbors to connect to
            level_0_neighbors_count: Number of neighbors at level 0
            
        Returns:
            Index object for the newly created index
        """
        data = {
            "name": self.name,
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
        
        url = f"{self.client.base_url}/collections/{self.name}/indexes/dense"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create index: {response.text}")
        
        return Index(self.client, self)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this collection.
        
        Returns:
            JSON response containing collection information
        """
        url = f"{self.client.base_url}/collections/{self.name}"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get collection info: {response.text}")
            
        return response.json()
