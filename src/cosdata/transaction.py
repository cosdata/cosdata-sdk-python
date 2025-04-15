import json
import requests
from typing import Dict, Any, List, Optional

from contextlib import contextmanager
 # For type hints if needed

class Transaction:
    """
    Class for managing transactions in the vector database.
    """

    def __init__(self, client, collection_name: str, index_type: str, batch_size: int = 20, isIDF: bool = False):
        """
        Initialize a Transaction object.

        Args:
            client: VectorDBClient instance.
            collection_name: Name of the collection.
            index_type: "dense" or "sparse".
            batch_size: Maximum number of items per batch.
            isIDF: Flag indicating if the transaction is for IDF sparse index (default: False).
        """
        self.client = client
        self.collection_name = collection_name
        self.index_type = index_type
        self.batch_size = batch_size
        self.transaction_id: Optional[str] = None
        self.isIDF = isIDF  # Indicates if the operation is for an IDF sparse index
        print(f"[DEBUG] Creating transaction for index type: {self.index_type}, isIDF: {self.isIDF}")
        self._create()
        print(f"[DEBUG] Transaction ID: {self.transaction_id}")

    def _create(self) -> str:
        """
        Creates a new transaction using the refactored endpoint.
        """
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions"
        data = {"index_type": self.index_type}
        print(f"[DEBUG] Creating transaction - URL: {url}, Data: {data}")
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        print(f"[DEBUG] Create Transaction - Status Code: {response.status_code}, Response Text: {response.text}")
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create transaction: {response.text}")
        result = response.json()
        self.transaction_id = result.get("transaction_id")
        if not self.transaction_id:
            raise Exception("Transaction ID missing in response")
        return self.transaction_id

    def _upsert_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Upserts a batch of items into the active transaction.
        """
        if not self.client.token:
            self.client.login()
        if not self.transaction_id:
            self._create()

        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/upsert"
        data = {"index_type": self.index_type}
        print(f"[DEBUG] _upsert_batch - self.index_type: {self.index_type}, self.isIDF: {self.isIDF}")

        # Check if it's an IDF sparse index operation
        if self.index_type == "sparse" and self.isIDF:
            print("[DEBUG] _upsert_batch - IDF sparse upsert")
            data["isIDF"] = True
            data["documents"] = batch  # Use "documents" key for IDF upsert
            data["index_type"] = "sparse"
            print(f"[DEBUG] Upserting IDF batch - URL: {url}, Data: {json.dumps(data, indent=2)}")
        else:
            # For dense or non-IDF sparse upserts use the "vectors" key
            print("[DEBUG] _upsert_batch - Non-IDF/dense upsert")
            data["vectors"] = batch
            print(f"[DEBUG] Upserting non-IDF/dense batch - URL: {url}, Data: {json.dumps(data, indent=2)}")

        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        print(f"[DEBUG] Upsert Batch - Status Code: {response.status_code}, Response Text: {response.text}")
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to upsert vectors/documents: {response.text}")

    def upsert(self, items: List[Dict[str, Any]]) -> 'Transaction':
        """
        Upserts items in batches.
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
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/commit"
        data = {"index_type": self.index_type}
        print(f"[DEBUG] Committing transaction {self.transaction_id} - URL: {url}, Data: {data}")
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
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
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/abort"
        data = {"index_type": self.index_type}
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        self.transaction_id = None
        return response.json() if response.content else None

    # === Context Manager Support ===

    def __enter__(self) -> 'Transaction':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"[DEBUG] Transaction exited with exception, aborting: {exc_val}")
            self.abort()
        else:
            print(f"[DEBUG] Transaction exiting, committing.")
            self.commit()
