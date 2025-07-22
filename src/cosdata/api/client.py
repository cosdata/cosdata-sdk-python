# client.py
import json
import requests
from typing import Optional, Dict, Any, List
from .collections import Collection
from .auth import Auth


class Client:
    """
    Main client for interacting with the Cosdata Vector Database API.

    This client provides a Pythonic, object-oriented interface for interacting with
    the Cosdata Vector Database API.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8443",
        username: str = "admin",
        password: str = "admin",
        verify: bool = False,
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
        self.verify_ssl = verify

        # Initialize authentication
        self.auth = Auth(username, password)
        self.auth.set_client_info(host, verify)

        self._session = None

    def _get_headers(self) -> dict:
        """
        Get the headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return self.auth.get_headers()

    def _ensure_session(self):
        """Ensure the session is initialized."""
        if self._session is None:
            # Initialize session here
            pass

    def create_collection(
        self,
        name: str,
        dimension: int = 1024,
        description: Optional[str] = None,
        dense_vector: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, Any]] = None,
        tf_idf_options: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """
        Create a new collection.

        Args:
            name: Name of the collection
            dimension: Dimension of vectors to be stored
            description: Optional description of the collection
            dense_vector: Optional dense vector configuration
            sparse_vector: Optional sparse vector configuration
            tf_idf_options: Optional TF-IDF configuration

        Returns:
            Collection object
        """
        self._ensure_session()

        url = f"{self.base_url}/collections"
        data = {
            "name": name,
            "description": description,
            "dense_vector": dense_vector or {"enabled": True, "dimension": dimension},
            "sparse_vector": sparse_vector or {"enabled": False},
            "tf_idf_options": tf_idf_options or {"enabled": False},
            "config": {"max_vectors": None, "replication_factor": None},
            "store_raw_text": False,
        }

        response = requests.post(
            url,
            headers=self._get_headers(),
            data=json.dumps(data),
            verify=self.verify_ssl,
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create collection: {response.text}")

        return Collection(self, name)

    def get_collection(self, name: str) -> Collection:
        """
        Get an existing collection.

        Args:
            name: Name of the collection

        Returns:
            Collection object
        """
        self._ensure_session()

        url = f"{self.base_url}/collections/{name}"
        response = requests.get(
            url, headers=self._get_headers(), verify=self.verify_ssl
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get collection: {response.text}")

        return Collection(self, name)

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections.

        Returns:
            List of collection information dictionaries
        """
        self._ensure_session()

        url = f"{self.base_url}/collections"
        response = requests.get(
            url, headers=self._get_headers(), verify=self.verify_ssl
        )

        if response.status_code != 200:
            raise Exception(f"Failed to list collections: {response.text}"
                            
        response_data = response.json()
        
        # Handle both cases: direct list or dictionary with collections key
        if isinstance(response_data, list):
            return response_data
        elif isinstance(response_data, dict):
            return response_data.get("collections", [])
        else:
            return []

    def collections(self) -> List[Collection]:
        """
        Get all collections as Collection objects.

        Returns:
            List of Collection objects
        """
        return [Collection(self, coll["name"]) for coll in self.list_collections()]

