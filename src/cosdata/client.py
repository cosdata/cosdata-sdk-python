import json
import requests
from typing import Dict, Any, Iterator
from cosdata.collection import Collection  # Assuming collection.py is in src/cosdata

class Client:
    """
    Main client for interacting with the updated Vector Database API.
    This version assumes that collection-specific endpoints (like search,
    upsert, and indexes) follow the new /collections/{collection_id}/... structure.
    """

    def __init__(self, host: str = "http://127.0.0.1:8443", token: str = None, verify_ssl: bool = True) -> None:
        """
        Initialize the client with the provided host, token, and SSL verification settings.
        """
        self.host = host
        self.base_url = f"{host}/vectordb"
        self.token = token
        self.verify_ssl = verify_ssl
        self.login()

    def _get_headers(self) -> Dict[str, str]:
        """
        Returns the request headers, including the authorization token if available.
        """
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def login(self) -> str:
        """
        Authenticate with the server and obtain an access token.
        """
        url = f"{self.host}/auth/create-session"
        data = {"username": "admin", "password": "admin"}  # Replace with actual credentials
        response = requests.post(url, headers=self._get_headers(), data=json.dumps(data), verify=self.verify_ssl)
        
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.text}")
        
        session = response.json()
        self.token = session.get("access_token")
        return self.token

    def create_collection(self, name: str, dimension: int, description: str) -> Collection:
        """
        Create a new collection for vectors.
        The endpoint for collection creation remains at /collections.
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
            "sparse_vector": {
                "enabled": True,
                "auto_create_index": False
            },
            "metadata_schema": None,
            "config": {"max_vectors": None, "replication_factor": None},
        }
        response = requests.post(url, headers=self._get_headers(), data=json.dumps(data), verify=self.verify_ssl)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create collection: {response.text}")
        
        return Collection(self, name, dimension)

    def list_collections(self) -> Iterator[Collection]:
        """
        Retrieve a list of all collections.
        """
        url = f"{self.base_url}/collections"
        response = requests.get(url, headers=self._get_headers(), verify=self.verify_ssl)
        
        if response.status_code != 200:
            raise Exception(f"Failed to list collections: {response.text}")
        
        for coll_data in response.json():
            if isinstance(coll_data, dict):
                name = coll_data.get("name")
                dimension = coll_data.get("dense_vector", {}).get("dimension", 1024)
            else:
                name = coll_data
                dimension = 1024  
            yield Collection(self, name, dimension)

    def collections(self) -> Iterator[Collection]:
        """
        Alias for list_collections to maintain backward compatibility.
        """
        return self.list_collections()
