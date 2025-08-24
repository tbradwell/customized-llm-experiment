"""Skeleton document data model."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class SkeletonDocument:
    """Represents a generated skeleton document template."""
    
    # Core identification
    id: str
    source_document_ids: List[str] = field(default_factory=list)
    
    # Document content
    template_path: str = ""
    content_blocks: Dict[int, List[str]] = field(default_factory=dict)
    delimiter_positions: Dict[int, str] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    algorithm_version: str = "1.0"
    
    # Processing statistics
    total_paragraphs_processed: int = 0
    total_clusters_found: int = 0
    blocks_generated: int = 0
    uncertain_blocks: int = 0
    
    @property
    def has_content(self) -> bool:
        """Check if document has content blocks."""
        return len(self.content_blocks) > 0
    
    @property
    def block_count(self) -> int:
        """Number of content blocks in document."""
        return len(self.content_blocks)
    
    def add_content_block(self, position: int, alternatives: List[str]) -> None:
        """Add a content block with alternatives."""
        self.content_blocks[position] = alternatives
        self.blocks_generated += 1
    
    def add_delimiter(self, position: int, delimiter: str) -> None:
        """Add a delimiter at specific position."""
        self.delimiter_positions[position] = delimiter
    
    def get_block_at_position(self, position: int) -> Optional[List[str]]:
        """Get content block at specific position."""
        return self.content_blocks.get(position)
    
    def get_delimiter_at_position(self, position: int) -> Optional[str]:
        """Get delimiter at specific position."""
        return self.delimiter_positions.get(position)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'source_document_ids': self.source_document_ids,
            'template_path': self.template_path,
            'content_blocks': self.content_blocks,
            'delimiter_positions': self.delimiter_positions,
            'created_at': self.created_at.isoformat(),
            'algorithm_version': self.algorithm_version,
            'total_paragraphs_processed': self.total_paragraphs_processed,
            'total_clusters_found': self.total_clusters_found,
            'blocks_generated': self.blocks_generated,
            'uncertain_blocks': self.uncertain_blocks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkeletonDocument':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            source_document_ids=data.get('source_document_ids', []),
            template_path=data.get('template_path', ''),
            content_blocks=data.get('content_blocks', {}),
            delimiter_positions=data.get('delimiter_positions', {}),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            algorithm_version=data.get('algorithm_version', '1.0'),
            total_paragraphs_processed=data.get('total_paragraphs_processed', 0),
            total_clusters_found=data.get('total_clusters_found', 0),
            blocks_generated=data.get('blocks_generated', 0),
            uncertain_blocks=data.get('uncertain_blocks', 0)
        )
