# transactions.py
import json
import requests
from typing import Dict, Any, List, Optional, Union, Self
from contextlib import contextmanager

class Transaction:
    """
    Represents a transaction in the vector database.
    """
    
    def __init__(self, client, collection_name: str):
        """
        Initialize a transaction.
        
        Args:
            client: Client instance
            collection_name: Name of the collection
        """
        self.client = client
        self.collection_name = collection_name
        self.transaction_id: Optional[str] = None
        self.batch_size = 200  # Maximum vectors per batch
        self._create()
    
    def _create(self) -> str:
        """
        Create a new transaction.
        
        Returns:
            Transaction ID
        """
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
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
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/upsert"
        data = {"vectors": batch}
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to upsert vectors: {response.text}")
    
    def upsert(self, vectors: List[Dict[str, Any]]) -> Self:
        """
        Upsert vectors into the transaction, automatically splitting into batches.
        
        Args:
            vectors: List of dictionaries containing vector data
            
        Returns:
            Self for method chaining
        """
        # Split vectors into batches of batch_size
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            self._upsert_batch(batch)
            
        return self
    
    def add_vector(
        self,
        vector_id: str,
        dense_values: Optional[List[float]] = None,
        sparse_indices: Optional[List[int]] = None,
        sparse_values: Optional[List[float]] = None,
        text: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Self:
        """
        Add a single vector to the transaction.
        
        Args:
            vector_id: Unique identifier for the vector
            dense_values: Dense vector values
            sparse_indices: Indices for sparse vector
            sparse_values: Values for sparse vector
            text: Text content
            document_id: Optional document identifier
            
        Returns:
            Self for method chaining
        """
        if not self.transaction_id:
            self._create()
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/vectors"
        data = {
            "id": vector_id,
            "document_id": document_id
        }
        
        if dense_values is not None:
            data["dense_values"] = dense_values
        if sparse_indices is not None and sparse_values is not None:
            data["sparse_indices"] = sparse_indices
            data["sparse_values"] = sparse_values
        if text is not None:
            data["text"] = text
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to add vector: {response.text}")
            
        return self
    
    def delete_vector(self, vector_id: str) -> Self:
        """
        Delete a vector from the transaction.
        
        Args:
            vector_id: ID of the vector to delete
            
        Returns:
            Self for method chaining
        """
        if not self.transaction_id:
            self._create()
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/vectors/{vector_id}"
        response = requests.delete(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to delete vector: {response.text}")
            
        return self
    
    def commit(self) -> None:
        """
        Commit the transaction.
        """
        if not self.transaction_id:
            raise Exception("No active transaction to commit")
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/commit"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
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
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/abort"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
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
                txn.upsert(vectors)
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