"""Clustering engine for skeleton processor algorithm steps 2-3."""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .models.paragraph import Paragraph
from .models.cluster import Cluster
from .utils.clustering_utils import SphericalKMeans, ClusterAnalyzer
from .repositories.cluster_repository import ClusterRepository
from .repositories.paragraph_repository import ParagraphRepository

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Handles paragraph clustering and homogeneity scoring."""
    
    def __init__(self, 
                 high_homogeneity_threshold: float = 0.8,
                 low_homogeneity_threshold: float = 0.8,
                 cluster_repository: Optional[ClusterRepository] = None,
                 paragraph_repository: Optional[ParagraphRepository] = None,
                 min_cluster_size_ratio: float = 0.25):
        """Initialize clustering engine.
        
        Args:
            high_homogeneity_threshold: Threshold for high homogeneity (>= this value)
            low_homogeneity_threshold: Threshold for low homogeneity (< this value)
            cluster_repository: Optional repository for cluster database operations
            paragraph_repository: Optional repository for paragraph database operations
            min_cluster_size_ratio: Minimum cluster size as ratio of number of documents
        """
        self.high_homogeneity_threshold = high_homogeneity_threshold
        self.low_homogeneity_threshold = low_homogeneity_threshold
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.cluster_repository = cluster_repository
        self.paragraph_repository = paragraph_repository
        self.use_database = paragraph_repository is not None
        self.cluster_analyzer = ClusterAnalyzer()
        
    def cluster_paragraphs(self, paragraphs: List[Paragraph]) -> Dict[int, Cluster]:
        """Cluster paragraphs using embeddings according to algorithm steps 2-3.
        
        Args:
            paragraphs: List of paragraphs with embeddings
            
        Returns:
            Dictionary mapping cluster IDs to Cluster objects
        """
        logger.info(f"Starting clustering of {len(paragraphs)} paragraphs")
        
        if not paragraphs:
            logger.warning("No paragraphs provided for clustering")
            return {}
        
        # Step 2: Cluster the paragraphs using the embedding
        clusters = self._perform_clustering(paragraphs)
        
        # Update paragraph embedding table with cluster id
        self._assign_cluster_ids_to_paragraphs(paragraphs, clusters)
        
        # Step 3: For each cluster in the cluster table add the cluster homogeneity score
        self._calculate_homogeneity_scores(paragraphs, clusters)
        
        # Find representative members (medoid + farthest members)
        self._find_representative_members(paragraphs, clusters)
        
        # Save clusters to database if available
        if self.cluster_repository is not None and clusters:
            saved_count = self.cluster_repository.save_clusters_batch(list(clusters.values()))
            logger.info(f"Saved {saved_count} clusters to database")
        
        logger.info(f"Clustering completed. Created {len(clusters)} clusters")
        
        return clusters
    
    def _perform_clustering(self, paragraphs: List[Paragraph]) -> Dict[int, Cluster]:
        """Perform Spherical K-means clustering on paragraph embeddings.
        
        Args:
            paragraphs: List of paragraphs with embeddings
            
        Returns:
            Dictionary of clusters
        """
        # Extract embeddings
        embeddings = np.array([p.embedding for p in paragraphs if p.has_embedding])
        valid_paragraphs = [p for p in paragraphs if p.has_embedding]
        
        if len(embeddings) == 0:
            logger.error("No valid embeddings found for clustering")
            return {}
        
        # Validate embeddings
        if embeddings.shape[1] == 0:
            logger.error("Embeddings have 0 features")
            return {}
        
        # Filter out any embeddings that are all zeros
        non_zero_mask = np.any(embeddings != 0, axis=1)
        if not np.any(non_zero_mask):
            logger.error("All embeddings are zero vectors")
            return {}
        
        if np.sum(non_zero_mask) < len(embeddings):
            logger.warning(f"Filtering out {len(embeddings) - np.sum(non_zero_mask)} zero embeddings")
            embeddings = embeddings[non_zero_mask]
            valid_paragraphs = [p for i, p in enumerate(valid_paragraphs) if non_zero_mask[i]]
        
        logger.info(f"Clustering {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        # Calculate optimal K based on average number of paragraphs across documents
        k = self._calculate_optimal_k(valid_paragraphs)
        
        # Ensure K is not larger than number of embeddings
        k = min(k, len(embeddings))
        
        if k < 1:
            logger.error(f"Invalid K value: {k}")
            return {}
        
        logger.info(f"Using K={k} for Spherical K-means clustering")
        
        # Perform clustering
        try:
            kmeans = SphericalKMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            logger.info(f"Clustering completed with inertia: {kmeans.inertia_:.4f}")
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
        
        # Create cluster objects
        clusters = {}
        for cluster_id in range(k):
            cluster = Cluster(
                id=cluster_id,
                paragraph_ids=[],
                centroid=kmeans.cluster_centers_[cluster_id] if kmeans.cluster_centers_ is not None else None
            )
            clusters[cluster_id] = cluster
        
        # Assign paragraphs to clusters
        for paragraph, label in zip(valid_paragraphs, cluster_labels):
            clusters[label].add_paragraph(paragraph.id)
        
        # Log cluster sizes
        for cluster_id, cluster in clusters.items():
            logger.debug(f"Cluster {cluster_id}: {cluster.size} paragraphs")
        
        return clusters
    
    def _calculate_optimal_k(self, paragraphs: List[Paragraph]) -> int:
        """Calculate optimal K as average number of paragraphs across documents.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            Optimal number of clusters
        """
        # Count documents
        document_ids = set(p.document_id for p in paragraphs)
        num_documents = len(document_ids)
        
        if num_documents == 0:
            return max(1, len(paragraphs) // 10)  # Fallback
        
        # Calculate average paragraphs per document
        avg_paragraphs_per_doc = len(paragraphs) / num_documents
        
        # Use average as K, with some constraints
        k = max(2, int(avg_paragraphs_per_doc))  # Minimum 2 clusters
        k = min(k, len(paragraphs) // 2)  # Maximum half the paragraphs
        k = min(k, 50)  # Maximum 50 clusters for practicality
        
        logger.info(f"Calculated K={k} from {num_documents} documents, "
                   f"avg {avg_paragraphs_per_doc:.1f} paragraphs/doc")
        
        return k
    
    def _assign_cluster_ids_to_paragraphs(self, paragraphs: List[Paragraph], 
                                        clusters: Dict[int, Cluster]) -> None:
        """Assign cluster IDs to paragraph objects and update database.
        
        Args:
            paragraphs: List of paragraphs to update
            clusters: Dictionary of clusters
        """
        # Create mapping from paragraph ID to cluster ID
        paragraph_to_cluster = {}
        for cluster_id, cluster in clusters.items():
            for paragraph_id in cluster.paragraph_ids:
                paragraph_to_cluster[paragraph_id] = cluster_id
        
        # Update paragraphs in memory and database
        updated_count = 0
        for paragraph in paragraphs:
            if paragraph.id in paragraph_to_cluster:
                cluster_id = paragraph_to_cluster[paragraph.id]
                
                # Update in-memory
                paragraph.cluster_id = cluster_id
                logger.debug(f"Assigned paragraph {paragraph.id} to cluster {cluster_id}")
                
                # Update database immediately if available
                if self.use_database:
                    try:
                        success = self.paragraph_repository.update_paragraph_cluster_assignment(
                            paragraph.id, cluster_id, None  # certainty_type will be set later
                        )
                        if success:
                            updated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to update cluster assignment for paragraph {paragraph.id}: {e}")
        
        if self.use_database:
            logger.info(f"Updated {updated_count} paragraph cluster assignments in database")
    
    def _calculate_homogeneity_scores(self, paragraphs: List[Paragraph], 
                                    clusters: Dict[int, Cluster]) -> None:
        """Calculate homogeneity scores for each cluster.
        
        Args:
            paragraphs: List of paragraphs with embeddings
            clusters: Dictionary of clusters to update
        """
        logger.info("Calculating homogeneity scores for clusters")
        
        # Create embeddings array and labels
        embeddings = np.array([p.embedding for p in paragraphs if p.has_embedding])
        labels = np.array([p.cluster_id for p in paragraphs if p.has_embedding and p.cluster_id is not None])
        
        if len(embeddings) == 0:
            logger.warning("No embeddings available for homogeneity calculation")
            return
        
        # Calculate homogeneity for each cluster
        for cluster_id, cluster in clusters.items():
            try:
                homogeneity_score = self.cluster_analyzer.calculate_homogeneity_score(
                    embeddings, labels, cluster_id
                )
                cluster.homogeneity_score = homogeneity_score
                
                logger.debug(f"Cluster {cluster_id}: homogeneity score = {homogeneity_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate homogeneity for cluster {cluster_id}: {e}")
                cluster.homogeneity_score = 0.0
        
        # Log homogeneity statistics
        scores = [c.homogeneity_score for c in clusters.values()]
        if scores:
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            logger.info(f"Homogeneity scores - Avg: {avg_score:.4f}, "
                       f"Min: {min_score:.4f}, Max: {max_score:.4f}")
    
    def _find_representative_members(self, paragraphs: List[Paragraph], 
                                   clusters: Dict[int, Cluster]) -> None:
        """Find medoid and farthest members for each cluster.
        
        Args:
            paragraphs: List of paragraphs with embeddings
            clusters: Dictionary of clusters to update
        """
        logger.info("Finding representative members for clusters")
        
        # Create embeddings array and labels
        embeddings = np.array([p.embedding for p in paragraphs if p.has_embedding])
        labels = np.array([p.cluster_id for p in paragraphs if p.has_embedding and p.cluster_id is not None])
        
        if len(embeddings) == 0:
            logger.warning("No embeddings available for representative member calculation")
            return
        
        # Create mapping from array index to paragraph ID
        index_to_paragraph_id = {
            i: p.id for i, p in enumerate(paragraphs) 
            if p.has_embedding and p.cluster_id is not None
        }
        
        for cluster_id, cluster in clusters.items():
            try:
                # Find medoid
                medoid_idx = self.cluster_analyzer.find_medoid(embeddings, labels, cluster_id)
                if medoid_idx >= 0 and medoid_idx in index_to_paragraph_id:
                    cluster.medoid_id = index_to_paragraph_id[medoid_idx]
                    logger.debug(f"Cluster {cluster_id}: medoid = {cluster.medoid_id}")
                
                # Find farthest members
                if cluster.medoid_id:
                    farthest_indices = self.cluster_analyzer.find_farthest_members(
                        embeddings, labels, cluster_id, medoid_idx, n_members=2
                    )
                    
                    cluster.farthest_member_ids = [
                        index_to_paragraph_id[idx] for idx in farthest_indices
                        if idx in index_to_paragraph_id
                    ]
                    
                    logger.debug(f"Cluster {cluster_id}: farthest members = {cluster.farthest_member_ids}")
                
            except Exception as e:
                logger.warning(f"Failed to find representative members for cluster {cluster_id}: {e}")
                cluster.medoid_id = None
                cluster.farthest_member_ids = []
    
    def assign_certainty_types(self, paragraphs: List[Paragraph], 
                             clusters: Dict[int, Cluster]) -> None:
        """Assign certainty types to paragraphs based on cluster homogeneity.
        
        Args:
            paragraphs: List of paragraphs to update
            clusters: Dictionary of clusters with homogeneity scores
        """
        logger.info("Assigning certainty types to paragraphs")
        
        certain_count = 0
        uncertain_count = 0
        
        for paragraph in paragraphs:
            if paragraph.cluster_id is not None and paragraph.cluster_id in clusters:
                cluster = clusters[paragraph.cluster_id]
                
                # Algorithm step 5.8-5.9: Assign type based on homogeneity
                if cluster.homogeneity_score >= self.high_homogeneity_threshold:
                    paragraph.certainty_type = "certain"
                    certain_count += 1
                else:
                    paragraph.certainty_type = "uncertain"
                    uncertain_count += 1
                
                logger.debug(f"Paragraph {paragraph.id}: {paragraph.certainty_type} "
                           f"(homogeneity: {cluster.homogeneity_score:.4f})")
            else:
                paragraph.certainty_type = "uncertain"  # Default for unassigned
                uncertain_count += 1
        
        logger.info(f"Assigned certainty types: {certain_count} certain, {uncertain_count} uncertain")
    
    def get_clustering_stats(self, paragraphs: List[Paragraph], 
                           clusters: Dict[int, Cluster]) -> Dict[str, any]:
        """Get statistics about the clustering results.
        
        Args:
            paragraphs: List of paragraphs
            clusters: Dictionary of clusters
            
        Returns:
            Dictionary with clustering statistics
        """
        if not clusters:
            return {}
        
        # Cluster size statistics
        cluster_sizes = [cluster.size for cluster in clusters.values()]
        
        # Homogeneity statistics
        homogeneity_scores = [cluster.homogeneity_score for cluster in clusters.values()]
        
        # Certainty statistics
        certain_count = sum(1 for p in paragraphs if p.certainty_type == "certain")
        uncertain_count = sum(1 for p in paragraphs if p.certainty_type == "uncertain")
        
        # Representative member statistics
        clusters_with_medoid = sum(1 for c in clusters.values() if c.medoid_id is not None)
        avg_farthest_members = np.mean([len(c.farthest_member_ids) for c in clusters.values()])
        
        return {
            'total_clusters': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'avg_homogeneity': np.mean(homogeneity_scores),
            'min_homogeneity': np.min(homogeneity_scores),
            'max_homogeneity': np.max(homogeneity_scores),
            'certain_paragraphs': certain_count,
            'uncertain_paragraphs': uncertain_count,
            'clusters_with_medoid': clusters_with_medoid,
            'avg_farthest_members_per_cluster': avg_farthest_members,
            'high_homogeneity_threshold': self.high_homogeneity_threshold,
            'low_homogeneity_threshold': self.low_homogeneity_threshold
        }
    
    def validate_clustering(self, paragraphs: List[Paragraph], 
                          clusters: Dict[int, Cluster]) -> Tuple[bool, List[str]]:
        """Validate clustering results.
        
        Args:
            paragraphs: List of paragraphs
            clusters: Dictionary of clusters
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not clusters:
            issues.append("No clusters created")
            return False, issues
        
        # Check that all paragraphs are assigned to clusters
        unassigned_paragraphs = [p for p in paragraphs if p.cluster_id is None]
        if unassigned_paragraphs:
            issues.append(f"{len(unassigned_paragraphs)} paragraphs not assigned to clusters")
        
        # Check cluster integrity
        for cluster_id, cluster in clusters.items():
            if cluster.size == 0:
                issues.append(f"Cluster {cluster_id} is empty")
            
            if cluster.homogeneity_score < 0 or cluster.homogeneity_score > 1:
                issues.append(f"Cluster {cluster_id} has invalid homogeneity score: {cluster.homogeneity_score}")
            
            # Check representative members
            if cluster.size >= 3 and not cluster.medoid_id:
                issues.append(f"Cluster {cluster_id} with {cluster.size} members has no medoid")
        
        # Check certainty type assignment
        untyped_paragraphs = [p for p in paragraphs if not p.certainty_type]
        if untyped_paragraphs:
            issues.append(f"{len(untyped_paragraphs)} paragraphs have no certainty type")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Clustering validation passed")
        else:
            logger.warning(f"Clustering validation failed: {issues}")
        
        return is_valid, issues
