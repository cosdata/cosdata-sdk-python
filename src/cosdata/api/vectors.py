# vectors.py
import json
import requests
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass


@dataclass
class Vector:
    """
    Represents a vector in the database.
    """

    id: str
    document_id: Optional[str]
    dense_values: Optional[List[float]]
    sparse_indices: Optional[List[int]]
    sparse_values: Optional[List[float]]
    text: Optional[str]


class Vectors:
    """
    Vectors module for managing vector operations.
    """

    def __init__(self, collection):
        """
        Initialize the vectors module.

        Args:
            collection: Collection instance
        """
        self.collection = collection

    def get(self, vector_id: str) -> Vector:
        """
        Get a vector by its ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Vector object
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/vectors/{vector_id}"
        response = requests.get(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get vector: {response.text}")

        data = response.json()
        return Vector(
            id=data["id"],
            document_id=data.get("document_id"),
            dense_values=data.get("dense_values"),
            sparse_indices=data.get("sparse_indices"),
            sparse_values=data.get("sparse_values"),
            text=data.get("text"),
        )

    def get_by_document_id(self, document_id: str) -> List[Vector]:
        """
        Get all vectors associated with a document ID.

        Args:
            document_id: Document ID to query vectors for

        Returns:
            List of Vector objects
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/vectors"
        params = {"document_id": document_id}
        response = requests.get(
            url,
            headers=self.collection.client._get_headers(),
            params=params,
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get vectors by document ID: {response.text}")

        data = response.json()
        return [
            Vector(
                id=vector_data["id"],
                document_id=vector_data.get("document_id"),
                dense_values=vector_data.get("dense_values"),
                sparse_indices=vector_data.get("sparse_indices"),
                sparse_values=vector_data.get("sparse_values"),
                text=vector_data.get("text"),
            )
            for vector_data in data
        ]

    def exists(self, vector_id: str) -> bool:
        """
        Check if a vector exists.

        Args:
            vector_id: ID of the vector to check

        Returns:
            True if the vector exists, False otherwise
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/vectors/{vector_id}"
        response = requests.head(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )

        return response.status_code == 200
