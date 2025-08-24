"""Cluster data model for skeleton processor."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Cluster:
    """Represents a cluster of similar paragraphs."""
    
    # Core identification
    id: int
    
    # Cluster data
    paragraph_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    homogeneity_score: float = 0.0
    
    # Representative members
    medoid_id: Optional[str] = None
    farthest_member_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if len(self.farthest_member_ids) > 2:
            # Keep only first 2 farthest members as per algorithm
            self.farthest_member_ids = self.farthest_member_ids[:2]
    
    @property
    def size(self) -> int:
        """Number of paragraphs in cluster."""
        return len(self.paragraph_ids)
    
    @property
    def has_enough_members(self) -> bool:
        """Check if cluster has enough members for medoid + 2 farthest."""
        return self.size >= 3
    
    @property
    def is_high_homogeneity(self) -> bool:
        """Check if cluster has high homogeneity (>= 0.8)."""
        return self.homogeneity_score >= 0.8
    
    @property
    def is_low_homogeneity(self) -> bool:
        """Check if cluster has low homogeneity (< 0.8)."""
        return self.homogeneity_score < 0.8
    
    def add_paragraph(self, paragraph_id: str) -> None:
        """Add a paragraph to the cluster."""
        if paragraph_id not in self.paragraph_ids:
            self.paragraph_ids.append(paragraph_id)
    
    def remove_paragraph(self, paragraph_id: str) -> None:
        """Remove a paragraph from the cluster."""
        if paragraph_id in self.paragraph_ids:
            self.paragraph_ids.remove(paragraph_id)
    
    def get_representative_members(self) -> List[str]:
        """Get medoid + farthest members as per algorithm step 5.6."""
        representatives = []
        
        if self.medoid_id:
            representatives.append(self.medoid_id)
        
        representatives.extend(self.farthest_member_ids)
        
        return representatives
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for database storage."""
        return {
            'id': self.id,
            'paragraph_ids': self.paragraph_ids,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'homogeneity_score': self.homogeneity_score,
            'medoid_id': self.medoid_id,
            'farthest_member_ids': self.farthest_member_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cluster':
        """Create cluster from dictionary."""
        cluster = cls(
            id=data['id'],
            paragraph_ids=data.get('paragraph_ids', []),
            homogeneity_score=data.get('homogeneity_score', 0.0),
            medoid_id=data.get('medoid_id'),
            farthest_member_ids=data.get('farthest_member_ids', [])
        )
        
        # Handle centroid
        if data.get('centroid') is not None:
            cluster.centroid = np.array(data['centroid'])
        
        return cluster
