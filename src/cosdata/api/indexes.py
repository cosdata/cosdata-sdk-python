# indexes.py
import json
import requests
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass


@dataclass
class DenseIndex:
    """
    Represents a dense vector index configuration.
    """

    name: str
    distance_metric_type: str
    quantization: Dict[str, Any]
    index: Dict[str, Any]


@dataclass
class SparseIndex:
    """
    Represents a sparse vector index configuration.
    """

    name: str
    quantization: int
    sample_threshold: int


@dataclass
class TfIdfIndex:
    """
    Represents a TF-IDF index configuration.
    """

    name: str
    sample_threshold: int
    k1: float
    b: float


class Index:
    """
    Represents an index in a collection.
    """

    def __init__(self, collection, name: str, index_type: str = "dense"):
        """
        Initialize an index.

        Args:
            collection: Collection instance
            name: Name of the index
            index_type: Type of index ("dense", "sparse", or "tf_idf")
        """
        self.collection = collection
        self.name = name
        self.index_type = index_type

    def delete(self) -> None:
        """
        Delete this index.
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/indexes/{self.index_type}"
        response = requests.delete(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code != 204:
            raise Exception(f"Failed to delete index: {response.text}")

