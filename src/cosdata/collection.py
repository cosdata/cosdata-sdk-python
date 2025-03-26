import json
import requests
import re
import sys
import unicodedata
import xxhash
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, List, Iterator, Set, Tuple
from .index import Index
from py_rust_stemmers import SnowballStemmer


def get_all_punctuation() -> Set[str]:
    return set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))

def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w]", " ", text.lower())
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()

def load_stopwords(stopwords_dir: Optional[Path], language: str) -> Set[str]:
    if stopwords_dir is None:
        
        stopwords_dir = Path(__file__).parent
       

    stopwords_path = stopwords_dir / "EN-Stopwords1.txt"
    if not stopwords_path.exists():
        print("Stopwords file not found. Using empty set.")
        return set()

    print(f"Stopwords file found at: {stopwords_path}")
    with stopwords_path.open("r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())
  
    return stopwords

def process_sentence(sentence: str,
                     language: str = "english",
                     token_max_length: int = 40,
                     disable_stemmer: bool = False,
                     stopwords_dir: Path = None) -> List[str]:
    custom_stopwords = load_stopwords(stopwords_dir, language)
    if custom_stopwords:
        print("Using custom stopwords from file.")
        stopwords = custom_stopwords
    else:
        print("Using built-in default stopwords.")
        stopwords = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "of", "for", "in", "to"}

    punctuation = get_all_punctuation()
    stemmer = SnowballStemmer(language) if not disable_stemmer else None

    cleaned = remove_non_alphanumeric(sentence)
    tokens = SimpleTokenizer.tokenize(cleaned)

    processed_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if token in punctuation:
            continue
        if lower_token in stopwords:
            continue
        if len(token) > token_max_length:
            continue
        stemmed_token = stemmer.stem_word(lower_token) if stemmer else lower_token
        if stemmed_token:
            processed_tokens.append(stemmed_token)
    return processed_tokens

def count_and_clamp_frequencies(tokens: List[str]) -> Dict[str, int]:
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    # clamp freq to max 8
    return {token: min(count, 8) for token, count in freq.items()}

def hash_token(token: str) -> int:
    return xxhash.xxh32(token.encode("utf-8")).intdigest() & 0xFFFFFFFF

def construct_sparse_vector(tokens: List[str]) -> Tuple[List[Tuple[int, np.uint8]], np.uint16]:
    freq = count_and_clamp_frequencies(tokens)
    sparse_vector = [(hash_token(token), np.uint8(count)) for token, count in freq.items()]
    doc_length = np.uint16(len(tokens))
    return sparse_vector, doc_length

##############################################
# Collection class with Hybrid Search
##############################################

class Collection:
    """
    Class for managing a collection in the vector database.
    """
    
    def __init__(self, client, name: str, dimension: int):
        """
        Initialize a Collection object.
        
        Args:
            client: VectorDBClient instance
            name: Name of the collection
            dimension: Dimensionality of vectors in this collection
        """
        self.client = client
        self.name = name
        self.dimension = dimension
    
    def index(self, distance_metric: str = "cosine") -> Index:
        """
        Get or create an index for this collection.
        
        Args:
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            
        Returns:
            Index object
        """
        return self.create_index(distance_metric=distance_metric)
    
    def create_index(self,
                     distance_metric: str = "cosine",
                     num_layers: int = 7,
                     max_cache_size: int = 1000,
                     ef_construction: int = 512,
                     ef_search: int = 256,
                     neighbors_count: int = 32,
                     level_0_neighbors_count: int = 64
    ) -> Index:
        """
        Create an index for this collection.
        
        Args:
            distance_metric: Type of distance metric (e.g., cosine, euclidean)
            num_layers: Number of layers in the HNSW graph
            max_cache_size: Maximum cache size
            ef_construction: ef parameter for index construction
            ef_search: ef parameter for search
            neighbors_count: Number of neighbors to connect to
            level_0_neighbors_count: Number of neighbors at level 0
            
        Returns:
            Index object for the newly created index
        """
        data = {
            "name": self.name,
            "distance_metric_type": distance_metric,
            "quantization": {"type": "auto", "properties": {"sample_threshold": 100}},
            "index": {
                "type": "hnsw",
                "properties": {
                    "num_layers": num_layers,
                    "max_cache_size": max_cache_size,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                    "neighbors_count": neighbors_count,
                    "level_0_neighbors_count": level_0_neighbors_count,
                },
            },
        }
        
        url = f"{self.client.base_url}/collections/{self.name}/indexes/dense"
        response = requests.post(
            url,
            headers=self.client._get_headers(),
            data=json.dumps(data),
            verify=self.client.verify_ssl
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create index: {response.text}")
        
        return Index(self.client, self)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this collection.
        
        Returns:
            JSON response containing collection information
        """
        url = f"{self.client.base_url}/collections/{self.name}"
        response = requests.get(
            url, 
            headers=self.client._get_headers(), 
            verify=self.client.verify_ssl
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get collection info: {response.text}")
            
        return response.json()

    ################################################################
    # HYBRID SEARCH: Combines Dense + Sparse in a single function
    ################################################################
    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both dense and sparse vectors.
        
        For dense search, a random dense vector is generated (replace with real embeddings in production).
        For sparse search, the query text is converted into a sparse vector.
        
        Args:
            query: Search query string.
            alpha: Weight for sparse search (0.0 - 1.0). 0.5 means equal weight.
            top_k: Number of top results to retrieve.
        
        Returns:
            List of documents with combined scores, sorted by relevance.
        """
        import numpy as np
        import json

        ##########################################################
        # 1) Dense Vector Search using global endpoint "/search"
        ##########################################################
        # Generate a random dense vector for the query
        dense_vector = np.random.uniform(-1, 1, self.dimension).tolist()  # Random dense vector
        dense_payload = {
            "vector_db_name": self.name,
            "vector": dense_vector,
            "nn_count": top_k
        }
        dense_response = requests.post(
            f"{self.client.base_url}/search",
            headers=self.client._get_headers(),
            data=json.dumps(dense_payload),
            verify=self.client.verify_ssl
        )
        if dense_response.status_code != 200:
            raise Exception(
                f"Dense search failed (status {dense_response.status_code}): {dense_response.text}"
            )
        dense_json = dense_response.json()
        if isinstance(dense_json, dict) and "RespVectorKNN" in dense_json:
            knn_list = dense_json["RespVectorKNN"].get("knn", [])
            dense_results = [{"id": item[0], "score": item[1].get("CosineSimilarity", 0)} for item in knn_list]
        elif isinstance(dense_json, list):
            if len(dense_json) > 0 and isinstance(dense_json[0], list):
                dense_results = dense_json[0]
            else:
                dense_results = dense_json
        else:
            dense_results = dense_json

        ##########################################################
        # 2) Sparse Vector Search
        ##########################################################
        sparse_tokens = process_sentence(query, language="english")
        sparse_vector, doc_length = construct_sparse_vector(sparse_tokens)
        sparse_vector_serializable = [[pair[0], int(pair[1])] for pair in sparse_vector]
        # Generate a random dense vector for the sparse branch as well
        random_dense_for_sparse = np.random.uniform(-1, 1, self.dimension).tolist()
        sparse_payload = {
            "vector_db_name": self.name,
            "vector": random_dense_for_sparse,  # Random dense vector to satisfy API schema
            "sparse_vector": {
                "indices": [pair[0] for pair in sparse_vector_serializable],
                "values": [pair[1] for pair in sparse_vector_serializable],
                "doc_length": int(doc_length)
            },
            "nn_count": top_k
        }
        sparse_response = requests.post(
            f"{self.client.base_url}/search",
            headers=self.client._get_headers(),
            data=json.dumps(sparse_payload),
            verify=self.client.verify_ssl
        )
        if sparse_response.status_code != 200:
            raise Exception(
                f"Sparse search failed (status {sparse_response.status_code}): {sparse_response.text}"
            )
        sparse_json = sparse_response.json()
        if isinstance(sparse_json, dict) and "RespVectorKNN" in sparse_json:
            knn_list = sparse_json["RespVectorKNN"].get("knn", [])
            sparse_results = [{"id": item[0], "score": item[1].get("CosineSimilarity", 0)} for item in knn_list]
        elif isinstance(sparse_json, list):
            if len(sparse_json) > 0 and isinstance(sparse_json[0], list):
                sparse_results = sparse_json[0]
            else:
                sparse_results = sparse_json
        else:
            sparse_results = sparse_json

        ##########################################################
        # 3) Merge Results with Weighted Scores
        ##########################################################
        result_dict = {}
        for sres in sparse_results:
            doc_id = sres.get("id")
            doc_score = sres.get("score") if sres.get("score") is not None else 0
            doc_content = sres.get("document")
            result_dict[doc_id] = {"score": alpha * doc_score, "document": doc_content}
        
        for dres in dense_results:
            doc_id = dres.get("id")
            doc_score = dres.get("score") if dres.get("score") is not None else 0
            doc_content = dres.get("document")
            if doc_id in result_dict:
                result_dict[doc_id]["score"] += (1 - alpha) * doc_score
            else:
                result_dict[doc_id] = {"score": (1 - alpha) * doc_score, "document": doc_content}
        
        merged = [
            {"id": doc_id, "score": info["score"], "document": info["document"]}
            for doc_id, info in result_dict.items()
        ]
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return merged
