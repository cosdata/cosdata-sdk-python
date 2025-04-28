# collections.py
import json
import requests
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass

@dataclass
class Collection:
    """
    Represents a collection in the vector database.
    """
    name: str
    description: Optional[str]
    dense_vector: Dict[str, Any]
    sparse_vector: Dict[str, Any]
    tf_idf_options: Dict[str, Any]
    config: Dict[str, Any]
    store_raw_text: bool

class Collections:
    """
    Collections module for managing vector collections.
    """
    
    def __init__(self, client):
        """
        Initialize the collections module.
        
        Args:
            client: Client instance
        """
        self.client = client
    
    def create(
        self, 
        name: str, 
        dimension: int = 1024, 
        description: Optional[str] = None,
        dense_vector: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, Any]] = None,
        tf_idf_options: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        store_raw_text: bool = False
    ) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Name of the collection
            dimension: Dimensionality of vectors to be stored
            description: Optional description of the collection
            dense_vector: Configuration for dense vector support
            sparse_vector: Configuration for sparse vector support
            tf_idf_options: Configuration for text search/BM25 support
            config: Collection-level configuration options
            store_raw_text: Whether to store raw text in addition to processed text
            
        Returns:
            Collection object
        """
        url = f"{self.client.base_url}/collections"
        data = {
            "name": name,
            "description": description,
            "dense_vector": dense_vector or {
                "enabled": True,
                "dimension": dimension
            },
            "sparse_vector": sparse_vector or {
                "enabled": False
            },
            "tf_idf_options": tf_idf_options or {
                "enabled": False
            },
            "config": config or {
                "max_vectors": None,
                "replication_factor": None
            },
            "store_raw_text": store_raw_text
        }
        
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            data=json.dumps(data), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create collection: {response.text}")
        
        return self.get(name)
    
    def get(self, name: str) -> Collection:
        """
        Get a collection by name.
        
        Args:
            name: Name of the collection
            
        Returns:
            Collection object
        """
        url = f"{self.client.base_url}/collections/{name}"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get collection: {response.text}")
        
        data = response.json()
        return Collection(
            name=data["name"],
            description=data.get("description"),
            dense_vector=data["dense_vector"],
            sparse_vector=data["sparse_vector"],
            tf_idf_options=data["tf_idf_options"],
            config=data["config"],
            store_raw_text=data["store_raw_text"]
        )
    
    def list(self) -> List[Collection]:
        """
        List all collections.
        
        Returns:
            List of Collection objects
        """
        url = f"{self.client.base_url}/collections"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to list collections: {response.text}")
        
        collections_data = response.json().get("collections", [])
        return [
            Collection(
                name=data["name"],
                description=data.get("description"),
                dense_vector=data.get("dense_vector", {}),
                sparse_vector=data.get("sparse_vector", {}),
                tf_idf_options=data.get("tf_idf_options", {}),
                config=data.get("config", {}),
                store_raw_text=data.get("store_raw_text", False)
            )
            for data in collections_data
        ]
    
    def delete(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Name of the collection to delete
        """
        url = f"{self.client.base_url}/collections/{name}"
        response = requests.delete(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 204:
            raise Exception(f"Failed to delete collection: {response.text}")
    
    def load(self, name: str) -> Collection:
        """
        Load a collection into memory.
        
        Args:
            name: Name of the collection to load
            
        Returns:
            Collection object
        """
        url = f"{self.client.base_url}/collections/{name}/load"
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to load collection: {response.text}")
        
        return self.get(name)
    
    def unload(self, name: str) -> str:
        """
        Unload a collection from memory.
        
        Args:
            name: Name of the collection to unload
            
        Returns:
            Success message
        """
        url = f"{self.client.base_url}/collections/{name}/unload"
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to unload collection: {response.text}")
        
        return response.text 