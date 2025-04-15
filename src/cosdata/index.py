import json
import requests
from typing import Dict, Any, Iterator, Union
from contextlib import contextmanager

from .transaction import Transaction


class Index:
    """
    Class for managing indexes in the vector database.
    """
    def __init__(self, client, collection, index_type: str = "dense"):
        """
        Args:
            client: API client instance.
            collection: Collection object this index belongs to.
            index_type: 'dense' or 'sparse'
        """
        self.client = client
        self.collection = collection
        self.index_type = index_type

    def create_transaction(self) -> Transaction:
        """
        Create a new transaction for this index.

        Returns:
            Transaction object
        """
        return Transaction(
            client=self.client,
            collection_name=self.collection.name,
            index_type=self.index_type
        )

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """
        Context-managed transaction.

        Auto-commits if no exception is raised.
        Aborts if an exception occurs.

        Example:
            with index.transaction() as txn:
                txn.upsert([...])
        """
        txn = self.create_transaction()
        try:
            yield txn
            txn.commit()
        except Exception:
            txn.abort()
            raise

    def query(self, vector: list, nn_count: int = 5) -> Dict[str, Any]:
        """
        Query nearest neighbors for a given vector using the new dedicated search endpoints.

        Args:
            vector: The query vector.
            nn_count: Number of neighbors to return.

        Returns:
            Search results as a dictionary.
        """
        if self.index_type == "dense":
            url = f"{self.client.base_url}/collections/{self.collection.name}/search/dense"
            payload = {
                "vector": vector,
                "k": nn_count
            }
        elif self.index_type == "sparse":
            url = f"{self.client.base_url}/collections/{self.collection.name}/search/sparse"
            # For sparse search, assume 'vector' is already a serializable list (e.g. [[token_id, score], ...]).
            payload = {
                "values": vector,
                "top_k": nn_count
            }
        else:
            raise Exception("Unsupported index type for query.")

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(payload),
            verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Failed to query vector: {response.text}")
        return response.json()

    def fetch_vector(self, vector_id: Union[str, int]) -> Dict[str, Any]:
        """
        Fetch a vector from the database by its ID using the updated endpoint.

        Args:
            vector_id: The ID of the vector to fetch.

        Returns:
            The vector's data.
        """
        # Updated endpoint: using the new vector route under the collection scope.
        url = f"{self.client.base_url}/collections/{self.collection.name}/vectors/{vector_id}"
        response = requests.get(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
        )
        if response.status_code != 200:
            raise Exception(f"Failed to fetch vector: {response.text}")
        return response.json()
