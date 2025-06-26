# transactions.py
import json
import requests
from typing import Dict, Any, List, Optional, Union, Self
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
        self.collection = collection_or_index if hasattr(collection_or_index, 'name') else collection_or_index.collection
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
            verify=self.collection.client.verify_ssl
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
            verify=self.collection.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to upsert vectors: {response.text}")
    
    def upsert_vector(self, vector: Dict[str, Any]) -> None:
        """
        Insert or update a single vector in the transaction.
        
        Args:
            vector: Vector dictionary to upsert
        """
        self._upsert_batch([vector])
    
    def batch_upsert_vectors(self, vectors: List[Dict[str, Any]], max_workers: Optional[int] = None, max_retries: int = 3) -> None:
        """
        Insert or update multiple vectors in the transaction, using multi-threading and retry logic.

        Args:
            vectors: List of vector dictionaries to upsert
            max_workers: Number of threads to use (default: all available CPU threads)
            max_retries: Number of times to retry a failed batch (default: 3)
        """
        # Split vectors into batches of batch_size
        batches = [vectors[i:i + self.batch_size] for i in range(0, len(vectors), self.batch_size)]
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
            raise Exception(f"Batch {batch_idx} failed after {max_retries} retries: {last_exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(upsert_with_retries, batch, idx): idx for idx, batch in enumerate(batches)}
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
            verify=self.collection.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to commit transaction: {response.text}")
            
        self.transaction_id = None
    
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
            verify=self.collection.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to abort transaction: {response.text}")
            
        self.transaction_id = None

class Transactions:
    """
    Transactions module for managing vector transactions.
    """
    
    def __init__(self, client):
        """
        Initialize the transactions module.
        
        Args:
            client: Client instance
        """
        self.client = client
    
    def create(self, collection_name: str) -> Transaction:
        """
        Create a new transaction for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Transaction object
        """
        return Transaction(self.client, collection_name)
    
    @contextmanager
    def transaction(self, collection_name: str):
        """
        Create a transaction with context management.
        
        This allows for automatic commit on success or abort on exception.
        
        Example:
            with client.transactions.transaction("my_collection") as txn:
                txn.upsert_vector(vector)  # For single vector
                txn.batch_upsert_vectors(vectors)  # For multiple vectors
                # Auto-commits on exit or aborts on exception
        
        Args:
            collection_name: Name of the collection
            
        Yields:
            Transaction object
        """
        txn = self.create(collection_name)
        try:
            yield txn
            txn.commit()
        except Exception:
            txn.abort()
            raise
    
    def get_status(self, collection_name: str, transaction_id: str) -> str:
        """
        Get the status of a transaction.
        
        Args:
            collection_name: Name of the collection
            transaction_id: ID of the transaction to check
            
        Returns:
            Transaction status string
        """
        url = f"{self.client.base_url}/collections/{collection_name}/transactions/{transaction_id}/status"
        response = requests.get(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get transaction status: {response.text}")
        
        result = response.json()
        return result['status']
    
    def poll_completion(self, collection_name: str, transaction_id: str, target_status: str = 'complete', 
                       max_attempts: int = 10, sleep_interval: float = 1.0) -> tuple[str, bool]:
        """
        Poll transaction status until it reaches the target status or max attempts are exceeded.
        
        Args:
            collection_name: Name of the collection
            transaction_id: Transaction ID to poll
            target_status: Target status to wait for (default: 'complete')
            max_attempts: Maximum number of polling attempts
            sleep_interval: Time to sleep between attempts in seconds
        
        Returns:
            tuple: (final_status, success_boolean)
        """
        for attempt in range(max_attempts):
            try:
                print(f"Attempt {attempt + 1}: Waiting for transaction {transaction_id} to complete...")
                
                # Get actual transaction status
                status = self.get_status(collection_name, transaction_id)
                
                if status == target_status:
                    print(f"Transaction {transaction_id} completed successfully")
                    return status, True
                
                if attempt < max_attempts - 1:
                    time.sleep(sleep_interval)
                    
            except Exception as e:
                print(f"Error polling transaction status: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(sleep_interval)
        
        print(f"Transaction {transaction_id} may not have completed within {max_attempts} attempts")
        return "unknown", False 