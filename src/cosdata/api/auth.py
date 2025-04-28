# auth.py
import json
import requests
from typing import Dict, Optional

class Auth:
    """
    Authentication module for the Cosdata Vector Database API.
    """
    
    def __init__(self, client, username: str, password: str):
        """
        Initialize the authentication module.
        
        Args:
            client: Client instance
            username: Username for authentication
            password: Password for authentication
        """
        self.client = client
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self.login()
    
    def login(self) -> str:
        """
        Authenticate with the server and obtain an access token.
        
        Returns:
            The access token string
        """
        url = f"{self.client.host}/auth/create-session"
        data = {"username": self.username, "password": self.password}
        response = requests.post(
            url, 
            headers=self.get_headers(), 
            data=json.dumps(data), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.text}")
        
        session = response.json()
        self.token = session["access_token"]
        return self.token
    
    def get_headers(self) -> Dict[str, str]:
        """
        Generate request headers with authentication token if available.
        
        Returns:
            Dictionary of HTTP headers
        """
        headers = {"Content-type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers 