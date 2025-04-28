# client.py
from typing import Optional
from .auth import Auth
from .collections import Collections
from .transactions import Transactions
from .search import Search
from .indexes import Indexes
from .vectors import Vectors
from .versions import Versions

class Client:
    """
    Main client for interacting with the Cosdata Vector Database API.
    
    This client provides access to all API functionality through organized modules
    that match the API documentation structure.
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
        self.verify_ssl = verify
        
        # Initialize authentication
        self.auth = Auth(self, username, password)
        
        # Initialize API modules
        self.collections = Collections(self)
        self.transactions = Transactions(self)
        self.search = Search(self)
        self.indexes = Indexes(self)
        self.vectors = Vectors(self)
        self.versions = Versions(self)
    
    def _get_headers(self) -> dict:
        """
        Get the headers for API requests.
        
        Returns:
            Dictionary of HTTP headers
        """
        return self.auth.get_headers() 