# index.py
import json
import requests
from typing import Dict, Any, List, Optional, Union, Iterator, TypeVar, Generic, ContextManager
from contextlib import contextmanager
from .transaction import Transaction

T = TypeVar('T')

class TransactionContextManager(Generic[T], ContextManager[T]):
    """Context manager for transactions"""
    def __init__(self, transaction: T):
        self.transaction = transaction
    
    def __enter__(self) -> T:
        return self.transaction
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            # No exception occurred, commit the transaction
            self.transaction.commit()
        else:
            # An exception occurred, abort the transaction
            self.transaction.abort()

class Index:
    """
    Class for managing indexes in the vector database.
    """
    
    def __init__(self, client, collection):
        """
        Initialize an Index object.
        
        Args:
            client: VectorDBClient instance
            collection: Collection object this index belongs to
        """
        self.client = client
        self.collection = collection
    
    def create_transaction(self) -> Transaction:
        """
        Create a new transaction for this index.
        
        Returns:
            Transaction object
        """
        return Transaction(self.client, self.collection.name)
    
    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """
        Create a transaction with context management.
        
        This allows for automatic commit on success or abort on exception.
        
        Example:
            with index.transaction() as txn:
                txn.upsert(vectors)
                # Auto-commits on exit or aborts on exception
        
        Yields:
            Transaction object
        """
        txn = self.create_transaction()
        try:
            yield txn
            txn.commit()
        except Exception:
            txn.abort()
            raise
    
    def query(self, vector: List[float], nn_count: int = 5) -> Dict[str, Any]:
        """
        Search for nearest neighbors of a vector.
        
        Args:
            vector: Vector to search for similar vectors
            nn_count: Number of nearest neighbors to return
            
        Returns:
            Search results
        """
        url = f"{self.client.base_url}/search"
        data = {
            "vector_db_name": self.collection.name,
            "vector": vector,
            "nn_count": nn_count
        }
        
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            data=json.dumps(data), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to search vector: {response.text}")
            
        return response.json()
    
    def fetch_vector(self, vector_id: Union[str, int]) -> Dict[str, Any]:
        """
        Fetch a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to fetch
            
        Returns:
            Vector data
        """   
        url = f"{self.client.base_url}/fetch"
        data = {
            "vector_db_name": self.collection.name,
            "vector_id": vector_id
        }
        
        response = requests.post(
            url, 
            headers=self.client._get_headers(), 
            data=json.dumps(data), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch vector: {response.text}")
            
        return response.json()
