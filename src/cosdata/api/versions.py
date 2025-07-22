# versions.py
import json
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Version:
    """
    Represents a collection version.
    """

    version_number: int
    vector_count: int


class Versions:
    """
    Versions module for managing collection versions.
    """

    def __init__(self, collection):
        """
        Initialize the versions module.

        Args:
            collection: Collection instance
        """
        self.collection = collection

    def list(self) -> Dict[str, Any]:
        """
        Get a list of all versions for a collection.

        Returns:
            Dictionary containing version information and current_version
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/versions"
        response = requests.get(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to list versions: {response.text}")
        return response.json()

    def get_current(self) -> Version:
        """
        Get the currently active version of a collection.

        Returns:
            Version object representing the current version
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/versions/current"
        response = requests.get(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to list versions: {response.text}")
        return response.json()
