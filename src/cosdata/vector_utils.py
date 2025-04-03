import json
import re
import sys
import unicodedata
import xxhash
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from py_rust_stemmers import SnowballStemmer


def get_all_punctuation() -> Set[str]:
    return set(chr(i) for i in range(sys.maxunicode)
               if unicodedata.category(chr(i)).startswith("P"))


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w]", " ", text.lower())
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()


HARDCODED_STOPWORDS = {
    "a", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "s",
    "such", "t", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "www"
}

def process_sentence(sentence: str,
                     language: str = "english",
                     token_max_length: int = 40,
                     disable_stemmer: bool = False) -> List[str]:
    """
    Process the input sentence into stemmed tokens with hardcoded stopword removal.
    """
    print("Using hardcoded stopwords.")
    stopwords = HARDCODED_STOPWORDS
    punctuation = get_all_punctuation()
    stemmer = SnowballStemmer(language) if not disable_stemmer else None
    cleaned = remove_non_alphanumeric(sentence)
    tokens = SimpleTokenizer.tokenize(cleaned)
    processed_tokens = []

    for token in tokens:
        lower_token = token.lower()
        if token in punctuation or lower_token in stopwords or len(token) > token_max_length:
            continue
        stemmed_token = stemmer.stem_word(lower_token) if stemmer else lower_token
        if stemmed_token:
            processed_tokens.append(stemmed_token)

    return processed_tokens

def bm25_term_frequencies(tokens: List[str],
                          k: float = 1.2,
                          b: float = 0.75,
                          avg_len: float = 128.0) -> Dict[str, float]:
    tf_map: Dict[str, float] = {}
    counter: defaultdict[str, int] = defaultdict(int)
    for token in tokens:
        counter[token] += 1
    doc_len = len(tokens)
    for token, count in counter.items():
        tf_value = count * (k + 1) / (count + k * (1 - b + b * doc_len / avg_len))
        tf_map[token] = float(tf_value)
    return tf_map


def hash_token(token: str) -> int:
    return xxhash.xxh32(token.encode("utf-8")).intdigest() & 0xFFFFFFFF


def construct_sparse_vector(tokens: List[str]) -> Tuple[List[Tuple[int, np.float32]], int]:
    tf_dict = bm25_term_frequencies(tokens, k=1.2, b=0.75, avg_len=128.0)
    sparse_vector = [(hash_token(token), np.float32(value))
                     for token, value in tf_dict.items()]
    return sparse_vector, len(tokens)
