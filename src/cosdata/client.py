import requests
import json

class Client(object):
    def __init__(self,host, base_url):
        self.host = host
        self.base_url = base_url
    
    def create_session(self):
        url = f"{self.host}/auth/create-session"
        data = {"username": "admin", "password": "admin"}
        response = requests.post(
            url, headers=self.generate_headers(), data=json.dumps(data), verify=False
        )
        session = response.json()
        global token
        token = session["access_token"]
        self.token = token
        return token

    def generate_headers(self):
        return {"Authorization": f"Bearer {self.token}", "Content-type": "application/json"}

    def create_db(self, vector_name, description, dimension: int):
        url = f"{self.base_url}/collections"
        data = {
            "name": vector_name,
            "description": description,
            "dense_vector": {
                "enabled": True,
                "auto_create_index": False,
                "dimension": dimension,
            },
            "sparse_vector": {"enabled": False, "auto_create_index": False},
            "metadata_schema": None,
            "config": {"max_vectors": None, "replication_factor": None},
        }
        response = requests.post(
            url, headers=self.generate_headers(), data=json.dumps(data), verify=False
        )
        return response.json()

    def create_index(self, index_name, distance_metric):
        data = {
            "name": index_name,  # name of the index
            "distance_metric_type": distance_metric,  # Type of distance metric (e.g., cosine, euclidean)
            "quantization": {
                # "type": "scalar",
                # "properties": {
                #     "data_type": "f32",
                #     "range": {
                #         "min": -1.0,
                #         "max": 1.0,
                #     },
                # },
                "type": "auto",
                "properties": {"sample_threshold": 100},
            },
            "index": {
                "type": "hnsw",
                "properties": {
                    "num_layers": 10,
                    "max_cache_size": 1000,
                    "ef_construction": 64,
                    "ef_search": 64,
                    "neighbors_count": 16,
                    "level_0_neighbors_count": 32,
                },
            },
        }
        response = requests.post(
            f"{self.base_url}/collections/{index_name}/indexes/dense",
            headers=self.generate_headers(),
            data=json.dumps(data),
            verify=False,
        )

        if response.status_code not in [200, 204]:
            raise Exception(
                f"Failed to create index: {response.status_code} ({response.text})"
            )

        return response.json()