# transactions.py
import json
import requests
from typing import Dict, Any, List, Optional, Union, Self
from contextlib import contextmanager

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
    
    def batch_upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Insert or update multiple vectors in the transaction.
        
        Args:
            vectors: List of vector dictionaries to upsert
        """
        # Split vectors into batches of batch_size
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            self._upsert_batch(batch)
    
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