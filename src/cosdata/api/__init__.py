"""
Cosdata Vector Database API Client

This package provides a Python client for interacting with the Cosdata Vector Database API.
The client is organized into modules that match the API documentation structure:

- Overview: Basic client setup and configuration
- Authentication: User authentication and session management
- Collections: Collection management operations
- Transactions: Transaction-based vector operations
- Search: Vector search operations
- Indexes: Index management operations
- Vectors: Vector retrieval and management
- Versions: Collection version management
"""

from .client import Client
from .auth import Auth
from .collections import Collections
from .transactions import Transactions
from .search import Search
from .indexes import Indexes
from .vectors import Vectors
from .versions import Versions

__all__ = [
    'Client',
    'Auth',
    'Collections',
    'Transactions',
    'Search',
    'Indexes',
    'Vectors',
    'Versions'
] 