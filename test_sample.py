import numpy as np
import random
from cosdata.client import Client

client = Client(
    host="http://127.0.0.1:8443",
    base_url="http://127.0.0.1:8443/vectordb"
)

def generate_random_vector_with_id(id, length):
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values}

def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()

def generate_perturbation(base_vector, idd, perturbation_degree, dimensions):
    perturbation = np.random.uniform(
        -perturbation_degree, perturbation_degree, dimensions
    )

    perturbed_values = np.array(base_vector["values"]) + perturbation
    clamped_values = np.clip(perturbed_values, -1, 1)

    perturbed_vector = {"id": idd, "values": clamped_values.tolist()}
    return perturbed_vector

vector_db_name = "testdb_sdk"
dimension=768
description="Test Cosdata SDK"

db = client.create_db(
    vector_name=vector_db_name,
    description=description,
    dimension=dimension
)

print(client.create_index(
    vector_db_name,
    distance_metric="cosine"
))

transaction_res = db.create_transaction(
    collection_name=vector_db_name
)

test_vec = generate_random_vector_with_id(
    id=(100*(dimension+random.randint(1,300))),
    length=dimension
)

batch_vectors = [test_vec]

print(db.upsert_in_transaction(
    collection_name=vector_db_name,
    transaction_id=transaction_res['transaction_id'],
    vectors=batch_vectors
))

print(db.commit_transaction(
    collection_name=vector_db_name,
    transaction_id=transaction_res['transaction_id']
))

query_vec = generate_perturbation(
    base_vector=test_vec,
    idd=(100*(dimension+random.randint(1,300))),
    perturbation_degree=0.95,
    dimensions=dimension
)

print(db.query_vector(
    idd=query_vec['id'],
    vector=query_vec['values'],
    top_k=5
))