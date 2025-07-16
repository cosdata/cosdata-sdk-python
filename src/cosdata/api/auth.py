# auth.py
import json
import requests
from typing import Dict, Optional


class Auth:
    """
    Authentication module for the Cosdata Vector Database API.
    """

    def __init__(self, username: str, password: str):
        """
        Initialize the authentication module.

        Args:
            username: Username for authentication
            password: Password for authentication
        """
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self._host: Optional[str] = None
        self._verify_ssl: Optional[bool] = None

    def set_client_info(self, host: str, verify_ssl: bool):
        """Set client information needed for authentication."""
        self._host = host
        self._verify_ssl = verify_ssl
        self.login()

    def login(self) -> str:
        """
        Authenticate with the server and obtain an access token.

        Returns:
            The access token string
        """
        if not self._host:
            raise Exception("Client information not set. Call set_client_info first.")

        url = f"{self._host}/auth/create-session"
        data = {"username": self.username, "password": self.password}
        response = requests.post(
            url,
            headers=self.get_headers(),
            data=json.dumps(data),
            verify=self._verify_ssl,
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

