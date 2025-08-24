"""Cluster repository for database operations."""

import logging
from typing import List, Optional, Dict, Any
import uuid

from .base_repository import BaseRepository
from ..models.cluster import Cluster

logger = logging.getLogger(__name__)


class ClusterRepository(BaseRepository):
    """Repository for cluster database operations with pgvector support."""
    
    def save_cluster(self, cluster: Cluster) -> bool:
        """Save cluster with centroid to database."""
        query = """
        INSERT INTO clusters (
            id, centroid, homogeneity_score, medoid_id, farthest_member_ids
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            centroid = EXCLUDED.centroid,
            homogeneity_score = EXCLUDED.homogeneity_score,
            medoid_id = EXCLUDED.medoid_id,
            farthest_member_ids = EXCLUDED.farthest_member_ids;
        """
        
        try:
            # Format centroid for pgvector
            centroid_str = self.format_vector_for_db(cluster.centroid.tolist()) if cluster.centroid is not None else None
            
            params = (
                cluster.id,
                centroid_str,
                cluster.homogeneity_score,
                cluster.medoid_id,
                cluster.farthest_member_ids
            )
            
            self.execute_query(query, params, fetch=False)
            logger.debug(f"Saved cluster {cluster.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cluster {cluster.id}: {e}")
            return False
    
    def save_clusters_batch(self, clusters: List[Cluster]) -> int:
        """Save multiple clusters in a single transaction."""
        if not clusters:
            return 0
        
        query = """
        INSERT INTO clusters (
            id, centroid, homogeneity_score, medoid_id, farthest_member_ids
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            centroid = EXCLUDED.centroid,
            homogeneity_score = EXCLUDED.homogeneity_score,
            medoid_id = EXCLUDED.medoid_id,
            farthest_member_ids = EXCLUDED.farthest_member_ids;
        """
        
        try:
            params_list = []
            for cluster in clusters:
                centroid_str = self.format_vector_for_db(cluster.centroid.tolist()) if cluster.centroid is not None else None
                
                # Check if medoid_id exists in paragraphs table, if not set to None
                medoid_id = cluster.medoid_id
                if medoid_id:
                    check_query = "SELECT EXISTS(SELECT 1 FROM paragraphs WHERE id = %s);"
                    result = self.execute_query(check_query, (medoid_id,))
                    logger.debug(f"Checking medoid {medoid_id}: result={result}")
                    if not result or not result[0]['exists']:
                        logger.warning(f"Medoid {medoid_id} not found in paragraphs, setting to NULL for cluster {cluster.id}")
                        medoid_id = None
                    else:
                        logger.debug(f"Medoid {medoid_id} found in paragraphs, keeping for cluster {cluster.id}")
                
                params = (
                    cluster.id,
                    centroid_str,
                    cluster.homogeneity_score,
                    medoid_id,
                    cluster.farthest_member_ids
                )
                params_list.append(params)
            
            self.execute_many(query, params_list)
            logger.info(f"Saved {len(clusters)} clusters in batch")
            return len(clusters)
            
        except Exception as e:
            logger.error(f"Failed to save clusters batch: {e}")
            return 0
    
    def get_cluster_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        query = "SELECT * FROM clusters WHERE id = %s;"
        
        try:
            results = self.execute_query(query, (cluster_id,))
            if results:
                return self._row_to_cluster(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get cluster {cluster_id}: {e}")
            return None
    
    def get_all_clusters(self) -> List[Cluster]:
        """Get all clusters."""
        query = "SELECT * FROM clusters ORDER BY id;"
        
        try:
            results = self.execute_query(query)
            return [self._row_to_cluster(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get all clusters: {e}")
            return []
    
    def get_clusters_by_homogeneity_range(self, min_score: float, max_score: float) -> List[Cluster]:
        """Get clusters within a homogeneity score range."""
        query = """
        SELECT * FROM clusters 
        WHERE homogeneity_score >= %s AND homogeneity_score <= %s 
        ORDER BY homogeneity_score DESC;
        """
        
        try:
            results = self.execute_query(query, (min_score, max_score))
            return [self._row_to_cluster(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get clusters by homogeneity range: {e}")
            return []
    
    def get_high_homogeneity_clusters(self, threshold: float = 0.8) -> List[Cluster]:
        """Get clusters with high homogeneity (certain clusters)."""
        return self.get_clusters_by_homogeneity_range(threshold, 1.0)
    
    def get_low_homogeneity_clusters(self, threshold: float = 0.8) -> List[Cluster]:
        """Get clusters with low homogeneity (uncertain clusters)."""
        return self.get_clusters_by_homogeneity_range(0.0, threshold)
    
    def update_cluster_homogeneity(self, cluster_id: int, homogeneity_score: float) -> bool:
        """Update cluster homogeneity score."""
        query = "UPDATE clusters SET homogeneity_score = %s WHERE id = %s;"
        
        try:
            self.execute_query(query, (homogeneity_score, cluster_id), fetch=False)
            logger.debug(f"Updated cluster {cluster_id} homogeneity score to {homogeneity_score}")
            return True
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster_id} homogeneity: {e}")
            return False
    
    def update_cluster_medoid(self, cluster_id: int, medoid_id: str) -> bool:
        """Update cluster medoid."""
        query = "UPDATE clusters SET medoid_id = %s WHERE id = %s;"
        
        try:
            self.execute_query(query, (medoid_id, cluster_id), fetch=False)
            logger.debug(f"Updated cluster {cluster_id} medoid to {medoid_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster_id} medoid: {e}")
            return False
    
    def update_cluster_farthest_members(self, cluster_id: int, farthest_member_ids: List[str]) -> bool:
        """Update cluster farthest members."""
        query = "UPDATE clusters SET farthest_member_ids = %s WHERE id = %s;"
        
        try:
            self.execute_query(query, (farthest_member_ids, cluster_id), fetch=False)
            logger.debug(f"Updated cluster {cluster_id} farthest members")
            return True
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster_id} farthest members: {e}")
            return False
    
    def find_similar_clusters(self, centroid: List[float], limit: int = 5) -> List[tuple]:
        """Find clusters with similar centroids."""
        centroid_str = self.format_vector_for_db(centroid)
        
        query = """
        SELECT *, (centroid <=> %s::vector) as distance
        FROM clusters
        WHERE centroid IS NOT NULL
        ORDER BY distance
        LIMIT %s;
        """
        
        try:
            results = self.execute_query(query, (centroid_str, limit))
            similar_clusters = []
            
            for row in results:
                cluster = self._row_to_cluster(row)
                distance = float(row['distance'])
                similar_clusters.append((cluster, distance))
            
            return similar_clusters
        except Exception as e:
            logger.error(f"Failed to find similar clusters: {e}")
            return []
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete cluster and update related paragraphs."""
        try:
            with self.get_cursor() as cursor:
                # Update paragraphs to remove cluster assignment
                cursor.execute(
                    "UPDATE paragraphs SET cluster_id = NULL WHERE cluster_id = %s;", 
                    (cluster_id,)
                )
                updated_paragraphs = cursor.rowcount
                
                # Delete cluster
                cursor.execute("DELETE FROM clusters WHERE id = %s;", (cluster_id,))
                deleted_clusters = cursor.rowcount
                
                if deleted_clusters > 0:
                    logger.info(f"Deleted cluster {cluster_id} and updated {updated_paragraphs} paragraphs")
                    return True
                else:
                    logger.warning(f"Cluster {cluster_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete cluster {cluster_id}: {e}")
            return False
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get statistics about clusters in the database."""
        try:
            with self.get_cursor(commit=False) as cursor:
                # Total clusters
                cursor.execute("SELECT COUNT(*) as total FROM clusters;")
                total_clusters = cursor.fetchone()['total']
                
                # Homogeneity statistics
                cursor.execute("""
                    SELECT 
                        AVG(homogeneity_score) as avg_homogeneity,
                        MIN(homogeneity_score) as min_homogeneity,
                        MAX(homogeneity_score) as max_homogeneity,
                        STDDEV(homogeneity_score) as stddev_homogeneity
                    FROM clusters;
                """)
                homogeneity_stats = cursor.fetchone()
                
                # Cluster size distribution (number of paragraphs per cluster)
                cursor.execute("""
                    SELECT 
                        c.id,
                        COUNT(p.id) as paragraph_count
                    FROM clusters c
                    LEFT JOIN paragraphs p ON c.id = p.cluster_id
                    GROUP BY c.id
                    ORDER BY paragraph_count DESC;
                """)
                cluster_sizes = [{"cluster_id": row['id'], "paragraph_count": row['paragraph_count']} 
                               for row in cursor.fetchall()]
                
                # High vs low homogeneity clusters
                cursor.execute("SELECT COUNT(*) as count FROM clusters WHERE homogeneity_score >= 0.8;")
                high_homogeneity_count = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM clusters WHERE homogeneity_score < 0.8;")
                low_homogeneity_count = cursor.fetchone()['count']
                
                return {
                    "total_clusters": total_clusters,
                    "homogeneity_stats": {
                        "average": round(float(homogeneity_stats['avg_homogeneity']), 4) if homogeneity_stats['avg_homogeneity'] else 0,
                        "minimum": round(float(homogeneity_stats['min_homogeneity']), 4) if homogeneity_stats['min_homogeneity'] else 0,
                        "maximum": round(float(homogeneity_stats['max_homogeneity']), 4) if homogeneity_stats['max_homogeneity'] else 0,
                        "standard_deviation": round(float(homogeneity_stats['stddev_homogeneity']), 4) if homogeneity_stats['stddev_homogeneity'] else 0
                    },
                    "cluster_distribution": {
                        "high_homogeneity_clusters": high_homogeneity_count,
                        "low_homogeneity_clusters": low_homogeneity_count,
                        "high_homogeneity_percentage": round((high_homogeneity_count / total_clusters * 100), 2) if total_clusters > 0 else 0
                    },
                    "cluster_sizes": cluster_sizes[:10]  # Top 10 largest clusters
                }
                
        except Exception as e:
            logger.error(f"Failed to get cluster statistics: {e}")
            return {}
    
    def get_clusters_with_paragraph_counts(self) -> List[Dict[str, Any]]:
        """Get clusters with their paragraph counts."""
        query = """
        SELECT 
            c.*,
            COUNT(p.id) as paragraph_count
        FROM clusters c
        LEFT JOIN paragraphs p ON c.id = p.cluster_id
        GROUP BY c.id, c.centroid, c.homogeneity_score, c.medoid_id, c.farthest_member_ids, c.created_at
        ORDER BY paragraph_count DESC;
        """
        
        try:
            results = self.execute_query(query)
            clusters_with_counts = []
            
            for row in results:
                cluster_data = {
                    "cluster": self._row_to_cluster(row),
                    "paragraph_count": row['paragraph_count']
                }
                clusters_with_counts.append(cluster_data)
            
            return clusters_with_counts
        except Exception as e:
            logger.error(f"Failed to get clusters with paragraph counts: {e}")
            return []
    
    def _row_to_cluster(self, row: Dict[str, Any]) -> Cluster:
        """Convert database row to Cluster object."""
        import numpy as np
        
        # Parse centroid from pgvector format
        centroid = None
        if row['centroid']:
            centroid_list = self.parse_vector_from_db(row['centroid'])
            centroid = np.array(centroid_list) if centroid_list else None
        
        # Get paragraph IDs for this cluster (would need separate query in practice)
        paragraph_ids = []  # This would be filled by a separate method if needed
        
        return Cluster(
            id=row['id'],
            paragraph_ids=paragraph_ids,
            centroid=centroid,
            homogeneity_score=row['homogeneity_score'],
            medoid_id=row['medoid_id'],
            farthest_member_ids=row['farthest_member_ids'] or []
        )
