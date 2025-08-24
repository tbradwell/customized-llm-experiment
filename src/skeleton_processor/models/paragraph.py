"""Paragraph data model for skeleton processor."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid
import numpy as np


@dataclass
class Paragraph:
    """Represents a paragraph with its metadata and processing information."""
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    
    # Text content
    original_text: str = ""
    clean_text: str = ""
    
    # Position information
    absolute_position: int = 0
    relative_position: float = 0.0
    
    # Embedding and clustering
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    
    # Block and type assignments
    block_assignment: Optional[str] = None  # e.g., "block_1" or None
    certainty_type: str = ""  # "certain" or "uncertain"
    
    # Formatting preservation
    font_style: Dict[str, Any] = field(default_factory=dict)
    
    # Processing flags
    is_block: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.block_assignment is not None:
            self.is_block = True
    
    @property
    def has_embedding(self) -> bool:
        """Check if paragraph has an embedding."""
        return self.embedding is not None
    
    @property
    def is_certain(self) -> bool:
        """Check if paragraph is marked as certain."""
        return self.certainty_type == "certain"
    
    @property
    def is_uncertain(self) -> bool:
        """Check if paragraph is marked as uncertain."""
        return self.certainty_type == "uncertain"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert paragraph to dictionary for database storage."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'original_text': self.original_text,
            'clean_text': self.clean_text,
            'absolute_position': self.absolute_position,
            'relative_position': self.relative_position,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'cluster_id': self.cluster_id,
            'block_assignment': self.block_assignment,
            'certainty_type': self.certainty_type,
            'font_style': self.font_style,
            'is_block': self.is_block
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paragraph':
        """Create paragraph from dictionary."""
        paragraph = cls(
            id=data.get('id', str(uuid.uuid4())),
            document_id=data.get('document_id', ''),
            original_text=data.get('original_text', ''),
            clean_text=data.get('clean_text', ''),
            absolute_position=data.get('absolute_position', 0),
            relative_position=data.get('relative_position', 0.0),
            cluster_id=data.get('cluster_id'),
            block_assignment=data.get('block_assignment'),
            certainty_type=data.get('certainty_type', ''),
            font_style=data.get('font_style', {}),
            is_block=data.get('is_block', False)
        )
        
        # Handle embedding
        if data.get('embedding') is not None:
            paragraph.embedding = np.array(data['embedding'])
        
        return paragraph
