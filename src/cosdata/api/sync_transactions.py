import json
import requests
from typing import Dict, Any, List, Union

class SyncTransactions:
    """
    Synchronous transactions module for managing vector operations with immediate execution.
    """
    
    def __init__(self, client):
        """
        Initialize the sync transactions module.
        
        Args:
            client: Client instance
        """
        self.client = client
    
    def stream_upsert(self, collection_name: str, vectors: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Upsert vectors in a collection using streaming sync transaction.
        Returns immediately with the result.
        
        Args:
            collection_name: Name of the collection
            vectors: Single vector dict or list of vector dicts to upsert
            
        Returns:
            Response data from the upsert operation
        """
        # Ensure vectors is a list
        if isinstance(vectors, dict):
            vectors = [vectors]
        
        url = f"{self.client.base_url}/collections/{collection_name}/streaming/upsert"
        data = {"vectors": vectors}
        
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201, 204]:
            raise Exception(f"Failed to streaming upsert vectors: {response.text}")
        
        return response.json() if response.content else {}
    
    def stream_delete(self, collection_name: str, vector_id: str) -> Dict[str, Any]:
        """
        Delete a vector from a collection using streaming sync transaction.
        Returns immediately with the result.
        
        Args:
            collection_name: Name of the collection
            vector_id: ID of the vector to delete (single only)
            
        Returns:
            Response data from the delete operation
        """
        url = f"{self.client.base_url}/collections/{collection_name}/streaming/vectors/{vector_id}"
        response = requests.delete(
            url,
            headers=self.client._get_headers(),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201, 204]:
            raise Exception(f"Failed to streaming delete vector: {response.text}")
        
        return response.json() if response.content else {}
