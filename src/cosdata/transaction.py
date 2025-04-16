import json
import requests
from typing import Dict, Any, List, Optional

class Transaction:
    """
    Class for managing transactions in the vector database.
    Supports 'dense', 'sparse', and 'tf_idf' index types.
    """

    def __init__(self, client, collection_name: str, index_type: str, batch_size: int = 20):
        """
        Initialize a Transaction object.

        Args:
            client: VectorDBClient instance.
            collection_name: Name of the collection.
            index_type: "dense", "sparse", or "tf_idf".
            batch_size: Maximum number of items per batch.
        """
        self.client = client
        self.collection_name = collection_name
        self.index_type = index_type
        self.batch_size = batch_size
        self.transaction_id: Optional[str] = None
        print(f"[DEBUG] Creating transaction for index type: {self.index_type}")
        self._create()
        print(f"[DEBUG] Transaction ID: {self.transaction_id}")

    def _create(self) -> str:
        """
        Creates a new transaction using the transactions endpoint.
        """
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions"
        data = {"index_type": self.index_type}
        print(f"[DEBUG] Creating transaction - URL: {url}, Data: {data}")
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            json=data,
            verify=self.client.verify_ssl
        )
        print(f"[DEBUG] Create Transaction - Status Code: {response.status_code}, Response Text: {response.text}")
        if response.status_code not in (200, 201):
            raise Exception(f"Failed to create transaction: {response.text}")

        result = response.json()
        self.transaction_id = result.get("transaction_id")
        if not self.transaction_id:
            raise Exception("Transaction ID missing in response")
        return self.transaction_id

    def _upsert_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Upserts a batch of items into the active transaction.
        Uses 'vectors' for dense/sparse and 'documents' for tf_idf.
        """
        if not getattr(self.client, 'token', None):
            self.client.login()
        if not self.transaction_id:
            self._create()

        url = (
            f"{self.client.base_url}/collections/{self.collection_name}"
            f"/transactions/{self.transaction_id}/upsert"
        )
        data: Dict[str, Any] = {"index_type": self.index_type}
        key = "documents" if self.index_type == "tf_idf" else "vectors"
        data[key] = batch
        print(f"[DEBUG] Upserting batch - URL: {url}, Data: {json.dumps(data, indent=2)}")

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            json=data,
            verify=self.client.verify_ssl
        )
        print(f"[DEBUG] Upsert Batch - Status Code: {response.status_code}, Response Text: {response.text}")
        if response.status_code not in (200, 204):
            raise Exception(f"Failed to upsert items: {response.text}")

    def upsert(self, items: List[Dict[str, Any]]) -> 'Transaction':
        """
        Upserts items in batches according to batch_size.
        """
        print(f"[DEBUG] Starting upsert of {len(items)} items in transaction {self.transaction_id}")
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            print(f"[DEBUG] Processing batch of size {len(batch)}")
            self._upsert_batch(batch)
        return self

    def commit(self) -> Optional[Dict[str, Any]]:
        """
        Commits the active transaction.
        """
        if not self.transaction_id:
            raise Exception("No active transaction to commit")
        url = (
            f"{self.client.base_url}/collections/{self.collection_name}"
            f"/transactions/{self.transaction_id}/commit"
        )
        data = {"index_type": self.index_type}
        print(f"[DEBUG] Committing transaction {self.transaction_id} - URL: {url}, Data: {data}")

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            json=data,
            verify=self.client.verify_ssl
        )
        print(f"[DEBUG] Commit Transaction - Status Code: {response.status_code}, Response Text: {response.text}")
        self.transaction_id = None
        return response.json() if response.content else None

    def abort(self) -> Optional[Dict[str, Any]]:
        """
        Aborts the active transaction.
        """
        if not self.transaction_id:
            raise Exception("No active transaction to abort")
        url = (
            f"{self.client.base_url}/collections/{self.collection_name}"
            f"/transactions/{self.transaction_id}/abort"
        )
        data = {"index_type": self.index_type}
        print(f"[DEBUG] Aborting transaction {self.transaction_id} - URL: {url}, Data: {data}")

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            json=data,
            verify=self.client.verify_ssl
        )
        print(f"[DEBUG] Abort Transaction - Status Code: {response.status_code}, Response Text: {response.text}")
        self.transaction_id = None
        return response.json() if response.content else None

    def __enter__(self) -> 'Transaction':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"[DEBUG] Transaction exited with exception, aborting: {exc_val}")
            self.abort()
        else:
            print(f"[DEBUG] Transaction exiting, committing.")
            self.commit()
