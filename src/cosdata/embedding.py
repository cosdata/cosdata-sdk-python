try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

def embed_texts(texts, model_name="BAAI/bge-small-en-v1.5"):
    """
    Generate embeddings for a list of texts using cosdata-fastembed.
    Args:
        texts (list of str): The texts to embed.
        model_name (str): The model to use (default: "BAAI/bge-small-en-v1.5").
    Returns:
        list: List of embedding vectors (as plain Python lists).
    Raises:
        ImportError: If cosdata-fastembed is not installed.
    """
    if TextEmbedding is None:
        raise ImportError(
            "cosdata-fastembed is not installed. Please install it with 'pip install cosdata-fastembed' or 'pip install cosdata-fastembed-gpu'."
        )
    model = TextEmbedding(model_name=model_name)
    embeddings = list(model.embed(texts))
    # Ensure all embeddings are plain Python lists
    return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
