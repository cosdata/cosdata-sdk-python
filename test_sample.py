import numpy as np
from cosdata.client import Client
from cosdata.index import Index

client = Client(
    host="http://127.0.0.1:8443",
    base_url="http://127.0.0.1:8443/vectordb"
)

def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()


vector_db_name = "testdb_sdk"
dimension=768
description="Test Cosdata SDK"

db = client.create_db(
    vector_name=vector_db_name,
    description=description,
    dimension=dimension
)

random_vectors = generate_random_vector(
    10,
    dimension,
    -1,
    1
)