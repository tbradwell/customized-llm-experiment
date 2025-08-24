"""OpenAI embedding client for skeleton processor."""

import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np
import openai
from openai import OpenAI
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for creating embeddings using OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize embedding client.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = 100  # Batch size for API efficiency
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = np.array(response.data[0].embedding)
            logger.debug(f"Created embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Split into batches for API efficiency
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Created embeddings for batch {i//self.batch_size + 1}, "
                          f"size: {len(batch)}")
                
                # Rate limiting
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch {i//self.batch_size + 1}: {e}")
                raise
        
        logger.info(f"Successfully created {len(embeddings)} embeddings")
        return embeddings
    
    def create_position_embedding(self, position: int, total_positions: int, 
                                embedding_dim: int = 3072) -> np.ndarray:
        """Create relative position embedding.
        
        Args:
            position: Absolute position of paragraph
            total_positions: Total number of paragraphs in document
            embedding_dim: Dimension of embedding to match text embeddings
            
        Returns:
            Position embedding vector
        """
        if total_positions == 0:
            relative_position = 0.0
        else:
            relative_position = position / total_positions
        
        # Create a simple position encoding
        # Use sinusoidal encoding similar to transformer positional encodings
        position_embedding = np.zeros(embedding_dim)
        
        for i in range(0, embedding_dim, 2):
            div_term = np.exp(i * -np.log(10000.0) / embedding_dim)
            position_embedding[i] = np.sin(relative_position * div_term)
            if i + 1 < embedding_dim:
                position_embedding[i + 1] = np.cos(relative_position * div_term)
        
        return position_embedding
    
    def concatenate_embeddings(self, text_embedding: np.ndarray, 
                             position_embedding: np.ndarray) -> np.ndarray:
        """Concatenate text and position embeddings.
        
        Args:
            text_embedding: Text embedding vector
            position_embedding: Position embedding vector
            
        Returns:
            Combined embedding vector
        """
        # For concatenation, we add them element-wise (as per algorithm step 1.2.5)
        # Ensure they have the same dimension
        min_dim = min(len(text_embedding), len(position_embedding))
        
        combined = text_embedding[:min_dim] + position_embedding[:min_dim]
        return combined
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        # text-embedding-3-large has 3072 dimensions
        # text-embedding-3-small has 1536 dimensions
        # text-embedding-ada-002 has 1536 dimensions
        
        model_dims = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536, 
            "text-embedding-ada-002": 1536
        }
        
        return model_dims.get(self.model, 1536)  # Default to 1536
    
    def validate_embeddings(self, embeddings: List[np.ndarray]) -> bool:
        """Validate that all embeddings have correct dimensions.
        
        Args:
            embeddings: List of embedding vectors to validate
            
        Returns:
            True if all embeddings are valid
        """
        if not embeddings:
            return True
        
        expected_dim = self.get_embedding_dimension()
        
        for i, embedding in enumerate(embeddings):
            if embedding.shape[0] != expected_dim:
                logger.error(f"Embedding {i} has incorrect dimension: "
                           f"{embedding.shape[0]}, expected: {expected_dim}")
                return False
        
        logger.info(f"All {len(embeddings)} embeddings validated successfully")
        return True
