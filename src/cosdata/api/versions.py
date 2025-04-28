# versions.py
import json
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Version:
    """
    Represents a collection version.
    """
    hash: str
    version_number: int
    timestamp: int
    vector_count: int
    
    @property
    def created_at(self) -> datetime:
        """
        Get the creation timestamp as a datetime object.
        
        Returns:
            datetime object representing when the version was created
        """
        return datetime.fromtimestamp(self.timestamp)

class Versions:
    """
    Versions module for managing collection versions.
    """
    
    def __init__(self, client):
        """
        Initialize the versions module.
        
        Args:
            client: Client instance
        """
        self.client = client
    
    def list(self, collection_name: str) -> Dict[str, Any]:
        """
        Get a list of all versions for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing version information and current hash
        """
        url = f"{self.client.base_url}/collections/{collection_name}/versions"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to list versions: {response.text}")
            
        return response.json()
    
    def get_current(self, collection_name: str) -> Version:
        """
        Get the currently active version of a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Version object representing the current version
        """
        url = f"{self.client.base_url}/collections/{collection_name}/versions/current"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get current version: {response.text}")
            
        data = response.json()
        return Version(
            hash=data["hash"],
            version_number=data["version_number"],
            timestamp=data["timestamp"],
            vector_count=data["vector_count"]
        )
    
    def get(self, collection_name: str, version_hash: str) -> Version:
        """
        Get a specific version by its hash.
        
        Args:
            collection_name: Name of the collection
            version_hash: Hash of the version to retrieve
            
        Returns:
            Version object representing the requested version
        """
        url = f"{self.client.base_url}/collections/{collection_name}/versions/{version_hash}"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get version: {response.text}")
            
        data = response.json()
        return Version(
            hash=data["hash"],
            version_number=data["version_number"],
            timestamp=data["timestamp"],
            vector_count=data["vector_count"]
        ) 