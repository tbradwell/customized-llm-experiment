"""Paragraph repository for database operations with vector embeddings."""

import logging
from typing import List, Optional, Dict, Any, Tuple
import uuid

from .base_repository import BaseRepository
from ..models.paragraph import Paragraph

logger = logging.getLogger(__name__)


class ParagraphRepository(BaseRepository):
    """Repository for paragraph database operations with pgvector support."""
    
    def save_paragraph(self, paragraph: Paragraph) -> bool:
        """Save paragraph with embedding to database."""
        query = """
        INSERT INTO paragraphs (
            id, document_id, original_text, clean_text, 
            absolute_position, relative_position, embedding,
            cluster_id, block_assignment, certainty_type, font_style
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            clean_text = EXCLUDED.clean_text,
            embedding = EXCLUDED.embedding,
            cluster_id = EXCLUDED.cluster_id,
            block_assignment = EXCLUDED.block_assignment,
            certainty_type = EXCLUDED.certainty_type;
        """
        
        try:
            # Format embedding for pgvector
            embedding_str = self.format_vector_for_db(paragraph.embedding) if paragraph.embedding is not None else None
            font_style_json = self.format_json_for_db(paragraph.font_style)
            
            params = (
                paragraph.id,
                paragraph.document_id,
                paragraph.original_text,
                paragraph.clean_text,
                paragraph.absolute_position,
                paragraph.relative_position,
                embedding_str,
                paragraph.cluster_id,
                paragraph.block_assignment,
                paragraph.certainty_type,
                font_style_json
            )
            
            self.execute_query(query, params, fetch=False)
            logger.debug(f"Saved paragraph {paragraph.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save paragraph {paragraph.id}: {e}")
            return False
    
    def save_paragraphs_batch(self, paragraphs: List[Paragraph]) -> int:
        """Save multiple paragraphs in a single transaction."""
        if not paragraphs:
            return 0
        
        query = """
        INSERT INTO paragraphs (
            id, document_id, original_text, clean_text, 
            absolute_position, relative_position, embedding,
            cluster_id, block_assignment, certainty_type, font_style
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            clean_text = EXCLUDED.clean_text,
            embedding = EXCLUDED.embedding,
            cluster_id = EXCLUDED.cluster_id,
            block_assignment = EXCLUDED.block_assignment,
            certainty_type = EXCLUDED.certainty_type;
        """
        
        try:
            params_list = []
            for paragraph in paragraphs:
                embedding_str = self.format_vector_for_db(paragraph.embedding) if paragraph.embedding is not None else None
                font_style_json = self.format_json_for_db(paragraph.font_style)
                
                params = (
                    paragraph.id,
                    paragraph.document_id,
                    paragraph.original_text,
                    paragraph.clean_text,
                    paragraph.absolute_position,
                    paragraph.relative_position,
                    embedding_str,
                    paragraph.cluster_id,
                    paragraph.block_assignment,
                    paragraph.certainty_type,
                    font_style_json
                )
                params_list.append(params)
            
            self.execute_many(query, params_list)
            logger.info(f"Saved {len(paragraphs)} paragraphs in batch")
            return len(paragraphs)
            
        except Exception as e:
            logger.error(f"Failed to save paragraphs batch: {e}")
            return 0
    
    def get_paragraph_by_id(self, paragraph_id: str) -> Optional[Paragraph]:
        """Get paragraph by ID."""
        query = "SELECT * FROM paragraphs WHERE id = %s;"
        
        try:
            results = self.execute_query(query, (paragraph_id,))
            if results:
                return self._row_to_paragraph(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get paragraph {paragraph_id}: {e}")
            return None
    
    def get_paragraphs_by_document_id(self, document_id: str) -> List[Paragraph]:
        """Get all paragraphs for a document."""
        query = """
        SELECT * FROM paragraphs 
        WHERE document_id = %s 
        ORDER BY absolute_position;
        """
        
        try:
            results = self.execute_query(query, (document_id,))
            return [self._row_to_paragraph(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get paragraphs for document {document_id}: {e}")
            return []
    
    def get_paragraphs_by_cluster_id(self, cluster_id: int) -> List[Paragraph]:
        """Get all paragraphs in a cluster."""
        query = """
        SELECT * FROM paragraphs 
        WHERE cluster_id = %s 
        ORDER BY document_id, absolute_position;
        """
        
        try:
            results = self.execute_query(query, (cluster_id,))
            return [self._row_to_paragraph(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get paragraphs for cluster {cluster_id}: {e}")
            return []
    
    def update_paragraph_cluster_assignment(self, paragraph_id: str, cluster_id: int, 
                                          certainty_type: str = None) -> bool:
        """Update paragraph's cluster assignment."""
        if certainty_type:
            query = """
            UPDATE paragraphs 
            SET cluster_id = %s, certainty_type = %s 
            WHERE id = %s;
            """
            params = (cluster_id, certainty_type, paragraph_id)
        else:
            query = "UPDATE paragraphs SET cluster_id = %s WHERE id = %s;"
            params = (cluster_id, paragraph_id)
        
        try:
            self.execute_query(query, params, fetch=False)
            logger.debug(f"Updated paragraph {paragraph_id} cluster assignment to {cluster_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update paragraph {paragraph_id} cluster assignment: {e}")
            return False
    
    def update_paragraph_block_assignment(self, paragraph_id: str, block_assignment: str) -> bool:
        """Update paragraph's block assignment."""
        query = "UPDATE paragraphs SET block_assignment = %s WHERE id = %s;"
        
        try:
            self.execute_query(query, (block_assignment, paragraph_id), fetch=False)
            logger.debug(f"Updated paragraph {paragraph_id} block assignment")
            return True
        except Exception as e:
            logger.error(f"Failed to update paragraph {paragraph_id} block assignment: {e}")
            return False
    
    def update_paragraphs_cluster_assignments(self, paragraphs: List[Paragraph]) -> int:
        """Update cluster assignments for multiple paragraphs in batch."""
        if not paragraphs:
            return 0
        
        query = """
        UPDATE paragraphs 
        SET cluster_id = %s, certainty_type = %s, block_assignment = %s
        WHERE id = %s;
        """
        
        try:
            params_list = []
            total_paragraphs = len(paragraphs)
            paragraphs_with_clusters = 0
            
            for paragraph in paragraphs:
                if paragraph.cluster_id is not None:  # Only update if cluster is assigned
                    paragraphs_with_clusters += 1
                    params = (
                        paragraph.cluster_id,
                        paragraph.certainty_type,
                        paragraph.block_assignment,
                        paragraph.id
                    )
                    params_list.append(params)
                    logger.debug(f"Will update paragraph {paragraph.id} with cluster_id={paragraph.cluster_id}")
            
            logger.info(f"Found {paragraphs_with_clusters} paragraphs with cluster assignments out of {total_paragraphs} total")
            
            if params_list:
                self.execute_many(query, params_list)
                logger.info(f"Updated cluster assignments for {len(params_list)} paragraphs")
                return len(params_list)
            else:
                logger.warning("No paragraphs had cluster assignments to update")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to update paragraph cluster assignments: {e}")
            return 0
    
    def find_similar_paragraphs(self, embedding: List[float], limit: int = 10, 
                               exclude_document_id: str = None) -> List[Tuple[Paragraph, float]]:
        """Find similar paragraphs using vector similarity search."""
        embedding_str = self.format_vector_for_db(embedding)
        
        base_query = """
        SELECT *, (embedding <=> %s::vector) as distance
        FROM paragraphs
        WHERE embedding IS NOT NULL
        """
        
        params = [embedding_str]
        
        if exclude_document_id:
            base_query += " AND document_id != %s"
            params.append(exclude_document_id)
        
        base_query += " ORDER BY distance LIMIT %s;"
        params.append(limit)
        
        try:
            results = self.execute_query(base_query, tuple(params))
            similar_paragraphs = []
            
            for row in results:
                paragraph = self._row_to_paragraph(row)
                distance = float(row['distance'])
                similar_paragraphs.append((paragraph, distance))
            
            return similar_paragraphs
        except Exception as e:
            logger.error(f"Failed to find similar paragraphs: {e}")
            return []
    
    def get_all_paragraphs_with_embeddings(self) -> List[Paragraph]:
        """Get all paragraphs that have embeddings."""
        query = """
        SELECT * FROM paragraphs 
        WHERE embedding IS NOT NULL 
        ORDER BY document_id, absolute_position;
        """
        
        try:
            results = self.execute_query(query)
            return [self._row_to_paragraph(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get paragraphs with embeddings: {e}")
            return []
    
    def get_paragraphs_without_clusters(self) -> List[Paragraph]:
        """Get paragraphs that haven't been assigned to clusters."""
        query = """
        SELECT * FROM paragraphs 
        WHERE cluster_id IS NULL AND embedding IS NOT NULL
        ORDER BY document_id, absolute_position;
        """
        
        try:
            results = self.execute_query(query)
            return [self._row_to_paragraph(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get unclustered paragraphs: {e}")
            return []
    
    def delete_paragraphs_by_document_id(self, document_id: str) -> int:
        """Delete all paragraphs for a document."""
        query = "DELETE FROM paragraphs WHERE document_id = %s;"
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (document_id,))
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} paragraphs for document {document_id}")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete paragraphs for document {document_id}: {e}")
            return 0
    
    def get_paragraph_statistics(self) -> Dict[str, Any]:
        """Get statistics about paragraphs in the database."""
        try:
            with self.get_cursor(commit=False) as cursor:
                # Total paragraphs
                cursor.execute("SELECT COUNT(*) as total FROM paragraphs;")
                total_paragraphs = cursor.fetchone()['total']
                
                # Paragraphs with embeddings
                cursor.execute("SELECT COUNT(*) as total FROM paragraphs WHERE embedding IS NOT NULL;")
                paragraphs_with_embeddings = cursor.fetchone()['total']
                
                # Paragraphs with cluster assignments
                cursor.execute("SELECT COUNT(*) as total FROM paragraphs WHERE cluster_id IS NOT NULL;")
                clustered_paragraphs = cursor.fetchone()['total']
                
                # Certainty type distribution
                cursor.execute("""
                    SELECT certainty_type, COUNT(*) as count 
                    FROM paragraphs 
                    WHERE certainty_type IS NOT NULL 
                    GROUP BY certainty_type;
                """)
                certainty_distribution = {row['certainty_type']: row['count'] for row in cursor.fetchall()}
                
                # Average text lengths
                cursor.execute("""
                    SELECT 
                        AVG(LENGTH(original_text)) as avg_original_length,
                        AVG(LENGTH(clean_text)) as avg_clean_length
                    FROM paragraphs;
                """)
                length_stats = cursor.fetchone()
                
                return {
                    "total_paragraphs": total_paragraphs,
                    "paragraphs_with_embeddings": paragraphs_with_embeddings,
                    "clustered_paragraphs": clustered_paragraphs,
                    "embedding_coverage": round((paragraphs_with_embeddings / total_paragraphs * 100), 2) if total_paragraphs > 0 else 0,
                    "clustering_coverage": round((clustered_paragraphs / total_paragraphs * 100), 2) if total_paragraphs > 0 else 0,
                    "certainty_type_distribution": certainty_distribution,
                    "average_original_text_length": round(float(length_stats['avg_original_length']), 2) if length_stats['avg_original_length'] else 0,
                    "average_clean_text_length": round(float(length_stats['avg_clean_length']), 2) if length_stats['avg_clean_length'] else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get paragraph statistics: {e}")
            return {}
    
    def _row_to_paragraph(self, row: Dict[str, Any]) -> Paragraph:
        """Convert database row to Paragraph object."""
        # Parse embedding from pgvector format
        embedding = self.parse_vector_from_db(row['embedding']) if row['embedding'] else []
        
        # Parse font style from JSONB
        font_style = self.parse_json_from_db(row['font_style']) if row['font_style'] else {}
        
        return Paragraph(
            id=row['id'],
            document_id=row['document_id'],
            original_text=row['original_text'],
            clean_text=row['clean_text'],
            absolute_position=row['absolute_position'],
            relative_position=row['relative_position'],
            embedding=embedding,
            cluster_id=row['cluster_id'],
            block_assignment=row['block_assignment'],
            certainty_type=row['certainty_type'],
            font_style=font_style
        )
