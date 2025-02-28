import numpy as np
import random
import cosdata.client 


client = cosdata.client.Client(
    host="http://127.0.0.1:8443"
)

def generate_random_vector_with_id(id, length):
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values}

vector_db_name = "testdb_sdk_2"
dimension=768
description="Test Cosdata SDK"

random_vector = generate_random_vector_with_id(
    id=random.randint(1,100), 
    length=dimension
)

batch_vector = [random_vector]

client.create_collection(
    name=vector_db_name,
    dimension=dimension,
    description=description
)

index = client.create_index(
    collection_name=vector_db_name,
    distance_metric="cosine"
)

txn = index.create_transaction(
    collection_name=vector_db_name
)
txn.upsert_vectors(
    vectors=batch_vector
)
txn.commit()

print(index.search_vector(
    collection_name=vector_db_name,
    vector=random_vector["values"],
    nn_count=2
))

print(client.get_collection(
    collection_name=vector_db_name
))

print(client.list_collections())