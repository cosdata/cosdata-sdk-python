import requests
import json

class Index:
    def __init__(self, base_url, index_name, host, token):
        self.__token = token
        self.base_url = base_url
        self.index_name = index_name
        self.host = host

    def generate_headers(self):
        return {"Authorization": f"Bearer {self.__token}", "Content-type": "application/json"}

    def upsert_vector(self, vectors):
        url = f"{self.base_url}/upsert"
        data = {"vector_db_name": self.index_name, "vectors": vectors}
        response = requests.post(
            url, headers=self.generate_headers(), data=json.dumps(data), verify=False
        )
        return response

    def query_vector(self, idd, vector, top_k):
        url = f"{self.base_url}/search"
        data = {
            "vector_db_name": self.index_name, 
            "vector": vector, 
            "nn_count": top_k
        }
        response = requests.post(
            url, headers=self.generate_headers(), data=json.dumps(data), verify=False
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise Exception(f"Failed to search vector: {response.status_code}")
        result = response.json()

        # Handle empty results gracefully
        if not result.get("RespVectorKNN", {}).get("knn"):
            return (idd, {"RespVectorKNN": {"knn": []}})
    
        return (idd, result)

    def create_transaction(self, collection_name):
        url = f"{self.base_url}/collections/{collection_name}/transactions"
        data = {"index_type": "dense"}
        response = requests.post(
            url, data=json.dumps(data), headers=self.generate_headers(), verify=False
        )
        return response.json()

    def upsert_in_transaction(self, collection_name, transaction_id, vectors):
        url = (
            f"{self.base_url}/collections/{collection_name}/transactions/{transaction_id}/upsert"
        )
        data = {"index_type": "dense", "vectors": vectors}
        print(f"Request URL: {url}")
        print(f"Request Vectors Count: {len(vectors)}")
        response = requests.post(
            url,
            headers=self.generate_headers(),
            data=json.dumps(data),
            verify=False,
            timeout=10000,
        )
        print(f"Response Status: {response.status_code}")
        if response.status_code not in [200, 204]:
            raise Exception(
                f"Failed to create vector: {response.status_code} ({response.text})"
            )

        return response.text

    def commit_transaction(self, collection_name, transaction_id):
        url = (
            f"{self.base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
        )
        data = {"index_type": "dense"}
        response = requests.post(
            url, data=json.dumps(data), headers=self.generate_headers(), verify=False
        )
        
        if response.status_code not in [200, 204]:
            print(f"Error response: {response.text}")
            raise Exception(f"Failed to commit transaction: {response.status_code}")
        
        return response.json() if response.text else None


    def abort_transaction(self, collection_name, transaction_id):
        url = (
            f"{self.base_url}/collections/{collection_name}/transactions/{transaction_id}/abort"
        )
        data = {"index_type": "dense"}
        response = requests.post(
            url, data=json.dumps(data), headers=self.generate_headers(), verify=False
        )
        return response.json()