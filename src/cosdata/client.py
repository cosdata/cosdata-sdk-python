import requests
import json
from .index import Index
from .transaction import Transaction
from typing import List, Optional, Dict, Any, Union

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
        """
        self.host = host
        self.base_url = f"{host}/vectordb"
        self.username = username
        self.password = password
        self.token = None
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
    
    def create_collection(
        self, 
        name: str, 
        dimension: int = 1024, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new collection (database) for vectors.
        
        Args:
            name: Name of the collection
            dimension: Dimensionality of vectors to be stored
            description: Optional description of the collection
            
        Returns:
            JSON response from the server
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
            "sparse_vector": {"enabled": False, "auto_create_index": False},
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
            
        return response.json()
    
    def list_collections(self):
        """
        Retrieve a list of all collections in the vector database.
        
        This method sends a GET request to the server to fetch all available 
        collections. The response includes details about each collection such as 
        name, description, and configuration.
        
        Returns:
            requests.Response: The HTTP response object containing the list of collections.
            The JSON data can be accessed via response.json() which returns a list of 
            collection objects.
            
        Raises:
            Exception: If the server returns a non-successful status code.
        """
        response = requests.get(
            f"{self.base_url}/collections",
            headers=self._get_headers(),
            verify=False
        )

        if response.status_code not in [200]:
            raise Exception(f"Failed to list collection: {response.text}")
        
        return response

    def get_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a specific collection.
        
        Args:
            collection_name: Name of the collection to retrieve
            
        Returns:
            JSON response containing collection information
        """
        url = f"{self.base_url}/collections/{collection_name}"
        response = requests.get(
            url, 
            headers=self._get_headers(), 
            verify=self.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get collection: {response.text}")
            
        return response.json()
    
    def create_index(
        self, 
        collection_name: str,
        distance_metric: str = "cosine"
    ) -> Index:
        """
        Create an index for the specified collection.
        
        Args:
            collection_name: Name of the collection to create an index for
            
        Returns:
            JSON response from the server
        """
        index = Index(self, collection_name)
        index.create(distance_metric)
        
        return index