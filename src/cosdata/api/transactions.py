# transactions.py
import json
import requests
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import concurrent.futures
import os
import time


class Transaction:
    """
    Represents a transaction in the vector database.
    """

    def __init__(self, collection_or_index):
        """
        Initialize a transaction.

        Args:
            collection_or_index: Collection or Index instance
        """
        self.collection = (
            collection_or_index
            if hasattr(collection_or_index, "name")
            else collection_or_index.collection
        )
        self._vectors = []
        self.transaction_id = None
        self.batch_size = 200  # Maximum vectors per batch
        self._create()

    def _create(self) -> str:
        """
        Create a new transaction.

        Returns:
            Transaction ID
        """
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/transactions"
        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create transaction: {response.text}")

        result = response.json()
        self.transaction_id = result["transaction_id"]
        return self.transaction_id

    def _upsert_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Upsert a single batch of vectors.

        Args:
            batch: List of vector dictionaries to upsert
        """
        if not self.transaction_id:
            self._create()

        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/transactions/{self.transaction_id}/upsert"
        data = {"vectors": batch}

        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            data=json.dumps(data),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to upsert vectors: {response.text}")

    def upsert_vector(self, vector: Dict[str, Any]) -> None:
        """
        Insert or update a single vector in the transaction.
        This method will create a new vector or update an existing one.
        Use create_vector() if you only want to create new vectors.

        Args:
            vector: Vector dictionary to upsert
        """
        self._upsert_batch([vector])

    def batch_upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Insert or update multiple vectors in the transaction, using multi-threading and retry logic.

        Args:
            vectors: List of vector dictionaries to upsert
            max_workers: Number of threads to use (default: all available CPU threads)
            max_retries: Number of times to retry a failed batch (default: 3)
        """
        # Split vectors into batches of batch_size
        batches = [
            vectors[i : i + self.batch_size]
            for i in range(0, len(vectors), self.batch_size)
        ]
        exceptions = []
        if max_workers is None:
            max_workers = os.cpu_count() or 4

        def upsert_with_retries(batch, batch_idx):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    self._upsert_batch(batch)
                    return  # Success
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.5 * attempt)  # Exponential backoff
            raise Exception(
                f"Batch {batch_idx} failed after {max_retries} retries: {last_exc}"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(upsert_with_retries, batch, idx): idx
                for idx, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    future.result()
                except Exception as exc:
                    exceptions.append(exc)

        if exceptions:
            raise Exception(f"One or more batches failed: {exceptions}")

    def commit(self) -> None:
        """
        Commit the transaction.
        """
        if not self.transaction_id:
            raise Exception("No active transaction to commit")

        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/transactions/{self.transaction_id}/commit"
        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to commit transaction: {response.text}")


    def abort(self) -> None:
        """
        Abort the transaction.
        """
        if not self.transaction_id:
            raise Exception("No active transaction to abort")

        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/transactions/{self.transaction_id}/abort"
        response = requests.post(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )

        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to abort transaction: {response.text}")

    def get_status(self) -> dict:
           """
        Get the status of this transaction (or another, if specified).
        
        Args:
            collection_name: Name of the collection (default: this transaction's collection)
            transaction_id: ID of the transaction to check (default: this transaction's ID)
            
        Returns:
            Transaction status string
            """
        if not self.transaction_id:
            raise Exception("No active transaction to get status for")
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/transactions/{self.transaction_id}/status"
        response = requests.get(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get transaction status: {response.text}")
        return response.json()

    def delete_vector(self, vector_id: str) -> None:
        """
        Delete a vector by ID in the transaction.
        """
        if not self.transaction_id:
            raise Exception("No active transaction to delete vector for")
        url = f"{self.collection.client.base_url}/collections/{self.collection.name}/transactions/{self.transaction_id}/vectors/{vector_id}"
        response = requests.delete(
            url,
            headers=self.collection.client._get_headers(),
            verify=self.collection.client.verify_ssl,
        )
        if response.status_code != 204:
            raise Exception(f"Failed to delete vector in transaction: {response.text}")

    def poll_completion(
        self,
        target_status: str = "complete",
        max_attempts: int = 10,
        sleep_interval: float = 1.0,
    ) -> tuple[str, bool]:
        """
        Poll transaction status until it reaches the target status or max attempts are exceeded.

        Args:
            target_status: Target status to wait for (default: 'complete')
            max_attempts: Maximum number of polling attempts
            sleep_interval: Time to sleep between attempts in seconds

        Returns:
            tuple: (final_status, success_boolean)
        """
        for attempt in range(max_attempts):
            try:
                print(
                    f"Attempt {attempt + 1}: Waiting for transaction {self.transaction_id} to complete..."
                )

                # Get actual transaction status
                status_data = self.get_status()
                status = status_data.get("status", "unknown")

                if status == target_status:
                    print(f"Transaction {self.transaction_id} completed successfully")
                    return status, True

                if attempt < max_attempts - 1:
                    time.sleep(sleep_interval)


            except Exception as e:
                print(f"Error polling transaction status: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(sleep_interval)

        print(
            f"Transaction {self.transaction_id} may not have completed within {max_attempts} attempts"
        )
        return "unknown", False

