# client.py
import requests
import json
from typing import List, Optional, Dict, Any, Union, Iterator
from .collection import Collection
from .vector_utils import process_sentence, construct_sparse_vector

class Client:
    """
    Main client for interacting with the Vector Database API.
    """
    
    def __init__(
        self, 
        host: str = "http://127.0.0.1:8443", 
        username: str = "admin", 
        password: str = "admin",
        verify: bool = False
    ) -> None:
        """
        Initialize the Vector DB client.
        
        Args:
            host: Host URL of the Vector DB server
            username: Username for authentication
            password: Password for authentication
            verify: Whether to verify SSL certificates
        """
        self.host = host
        self.base_url = f"{host}/vectordb"
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self.verify_ssl = verify
        self.login()
    
    def login(self) -> str:
        """
        Authenticate with the server and obtain an access token.
        
        Returns:
            The access token string
        """
        url = f"{self.host}/auth/create-session"
        data = {"username": self.username, "password": self.password}
        response = requests.post(
            url, 
            headers=self._get_headers(), 
            data=json.dumps(data), 
            verify=self.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.text}")
        
        session = response.json()
        self.token = session["access_token"]
        return self.token
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Generate request headers with authentication token if available.
        
        Returns:
            Dictionary of HTTP headers
        """
        headers = {"Content-type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def collection(self, name: str) -> Collection:
        """
        Get a collection by name.
        
        Args:
            name: Name of the collection
            
        Returns:
            Collection object for the requested collection
        """
        return self.get_collection(name)
    
    def create_collection(
        self, 
        name: str, 
        dimension: int = 1024, 
        description: Optional[str] = None
    ) -> Collection:
        """
        Create a new collection (database) for vectors.
        
        Args:
            name: Name of the collection
            dimension: Dimensionality of vectors to be stored
            description: Optional description of the collection
            
        Returns:
            Collection object for the newly created collection
        """
        url = f"{self.base_url}/collections"
        data = {
            "name": name,
            "description": description,
            "dense_vector": {
                "enabled": True,
                "auto_create_index": False,
                "dimension": dimension,
            },
            "sparse_vector": {"enabled": True, "auto_create_index": True},
            "metadata_schema": None,
            "config": {"max_vectors": None, "replication_factor": None},
        }
        
        response = requests.post(
            url, 
            headers=self._get_headers(), 
            data=json.dumps(data), 
            verify=self.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create collection: {response.text}")
        
        # Return a Collection object for the newly created collection
        return Collection(self, name, dimension)
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get an existing collection.
        
        Args:
            collection_name: Name of the collection to retrieve
            
        Returns:
            Collection object for the requested collection
        """
        url = f"{self.base_url}/collections/{collection_name}"
        response = requests.get(
            url, 
            headers=self._get_headers(), 
            verify=self.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get collection: {response.text}")
        
        collection_info = response.json()
        dimension = collection_info.get("dense_vector", {}).get("dimension", 1024)
        
        return Collection(self, collection_name, dimension)
    
    def list_collections(self) -> requests.Response:
        """
        Retrieve a list of all collections in the vector database.
        
        Returns:
            HTTP response object containing the list of collections.
        """
        response = requests.get(
            f"{self.base_url}/collections",
            headers=self._get_headers(),
            verify=self.verify_ssl
        )

        if response.status_code not in [200]:
            raise Exception(f"Failed to list collections: {response.text}")
        
        return response
    
    def collections(self) -> Iterator[Collection]:
        """
        Iterator over all collections.
        
        Returns:
            Iterator of Collection objects
        """
        response = self.list_collections()
        for collection_data in response.json():
            name = collection_data.get("name")
            dimension = collection_data.get("dense_vector", {}).get("dimension", 1024)
            yield Collection(self, name, dimension)
