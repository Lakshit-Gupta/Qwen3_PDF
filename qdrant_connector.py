"""
Qdrant vector database connector for EduPlan AI.
This module provides a connector for interacting with the Qdrant vector database.
"""

import logging
import httpx
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QdrantConnector:
    """Connector for interacting with the Qdrant vector database."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "eduplan", vector_size: int = 4096):
        """
        Initialize the Qdrant connector.
        
        Args:
            host: Qdrant server hostname
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_size: Dimensionality of the vectors to store
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize client
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.debug(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
            
    def recreate_collection(self) -> bool:
        """
        Delete collection if it exists and create a new one.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name in collection_names:
                # Delete existing collection
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"🗑️ Deleted existing collection: {self.collection_name}")
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"✅ Created new collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")
            return False
            
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            return self.client.get_collection(collection_name=self.collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
            
    def insert_documents(self, documents: List[Dict], embeddings: List[List[float]], batch_size: int = 2) -> bool:
        """
        Insert documents with embeddings into Qdrant.
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and 'metadata'
            embeddings: List of embedding vectors (must match documents length)
            batch_size: Number of documents to insert at once
            
        Returns:
            True if insertion was successful
        """
        try:
            if len(documents) != len(embeddings):
                logger.error(f"Document count ({len(documents)}) does not match embeddings count ({len(embeddings)})")
                return False
                
            logger.info(f"Inserting {len(documents)} documents into collection '{self.collection_name}'")
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                # Prepare points for insertion
                points = []
                for doc, emb in zip(batch_docs, batch_embeddings):
                    # Ensure ID is a string or integer (not a list)
                    doc_id = doc.get("id")
                    if isinstance(doc_id, list):
                        # If ID is a list, convert to string
                        doc_id = str(doc_id)
                    
                    point = {
                        "id": doc_id,
                        "vector": emb,
                        "payload": {
                            "text": doc.get("text", ""),
                            "metadata": doc.get("metadata", {})
                        }
                    }
                    points.append(point)
                
                # Insert batch
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
            logger.info(f"Successfully inserted {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            return False
            
    def search_documents(self, query_vector: List[float], limit: int = 5, 
                        filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the collection.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of matching documents
        """
        try:
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter
            )
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
            
    def delete_document(self, document_id: Union[str, int]) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[document_id]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
            
    def get_document(self, document_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Get a document from the collection by ID.
        
        Args:
            document_id: ID of the document to get
            
        Returns:
            Document if found, None otherwise
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id]
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Test the connector
    connector = QdrantConnector(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        vector_size=4
    )
    
    # Create a test collection
    connector.recreate_collection()
    
    # Insert test documents
    test_docs = [
        {
            "id": 1,
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {"text": "Test document 1", "metadata": {"source": "test"}}
        },
        {
            "id": 2,
            "vector": [0.2, 0.3, 0.4, 0.5],
            "payload": {"text": "Test document 2", "metadata": {"source": "test"}}
        }
    ]
    
    connector.insert_documents(test_docs)
    
    # Search for similar documents
    results = connector.search_documents([0.1, 0.2, 0.3, 0.4], limit=1)
    print(f"Search results: {results}")

