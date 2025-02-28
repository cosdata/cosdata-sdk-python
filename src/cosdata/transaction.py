# transaction.py
import json
import requests
from typing import Dict, Any, List, Optional, Union, Self

class Transaction:
    """
    Class for managing transactions in the vector database.
    """
    
    def __init__(self, client, collection_name: str):
        """
        Initialize a Transaction object.
        
        Args:
            client: VectorDBClient instance
            collection_name: Name of the collection
            batch_size: Maximum number of vectors per batch
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
        data = {"index_type": "dense"}
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
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
        if not self.client.token:
            self.client.login()
            
        if not self.transaction_id:
            self._create()
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/upsert"
        data = {"index_type": "dense", "vectors": batch}
        
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
            vectors: List of dictionaries containing vector data with 'id' and 'values' keys
            
        Returns:
            Self for method chaining
        """
        # Split vectors into batches of batch_size
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            self._upsert_batch(batch)
            
        return self
    
    def commit(self) -> Optional[Dict[str, Any]]:
        """
        Commit the transaction.
        
        Returns:
            JSON response from the server or None
        """
        if not self.transaction_id:
            raise Exception("No active transaction to commit")
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/commit"
        data = {"index_type": "dense"}
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to commit transaction: {response.text}")
            
        self.transaction_id = None
        return response.json() if response.text else None
    
    def abort(self) -> Optional[Dict[str, Any]]:
        """
        Abort the transaction.
        
        Returns:
            JSON response from the server or None
        """
        if not self.transaction_id:
            raise Exception("No active transaction to abort")
            
        url = f"{self.client.base_url}/collections/{self.collection_name}/transactions/{self.transaction_id}/abort"
        data = {"index_type": "dense"}
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Failed to abort transaction: {response.text}")
            
        self.transaction_id = None
        return response.json() if response.text else None
