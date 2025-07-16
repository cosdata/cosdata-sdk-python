"""
Cosdata Vector Database API Client

This package provides a Python client for interacting with the Cosdata Vector Database API.
The client uses an object-oriented, chainable interface for a more intuitive experience.

The main components are:
- Client: Main entry point for the SDK
- Collection: Represents a collection in the database
- Index: Represents an index in a collection
- Transaction: Handles batch vector operations and status polling
- Search: Provides vector search functionality
- Vectors: Manages vector operations
- Versions: Manages collection versions
"""

from .client import Client
from .collections import Collection
from .indexes import Index
from .transactions import Transaction
from .search import Search
from .vectors import Vectors
from .versions import Versions

__all__ = [
    "Client",
    "Collection",
    "Index",
    "Transaction",
    "Search",
    "Vectors",
    "Versions",
]

