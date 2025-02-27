from typing import Dict, Any
import requests
import json


class Index:
    """
    Class for managing indexes in the vector database.
    """
    
    def __init__(
        self, 
        client, 
        collection_name: str
    ):
        """
        Initialize an Index object.
        
        Args:
            client: VectorDBClient instance
            collection_name: Name of the collection this index belongs to
        """
        self.client = client
        self.collection_name = collection_name
    
    def create(
        self, 
        distance_metric: str = "cosine",
        num_layers: int = 7,
        max_cache_size: int = 1000,
        ef_construction: int = 512,
        ef_search: int = 256,
        neighbors_count: int = 32,
        level_0_neighbors_count: int = 64
    ) -> Dict[str, Any]:
        """
        Create an index for the collection.
        
        Args:
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            num_layers: Number of layers in the HNSW graph
            max_cache_size: Maximum cache size
            ef_construction: ef parameter for index construction
            ef_search: ef parameter for search
            neighbors_count: Number of neighbors to connect to
            level_0_neighbors_count: Number of neighbors at level 0
            
        Returns:
            JSON response from the server
        """        
        data = {
            "name": self.collection_name,
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
        
        url = f"{self.client.base_url}/collections/{self.collection_name}/indexes/dense"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create index: {response.text}")
            
        return response.json()