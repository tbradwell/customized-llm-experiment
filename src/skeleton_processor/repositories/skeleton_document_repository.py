"""Skeleton document repository for database operations."""

import logging
from typing import List, Optional, Dict, Any
import json

from .base_repository import BaseRepository
from ..models.skeleton_document import SkeletonDocument

logger = logging.getLogger(__name__)


class SkeletonDocumentRepository(BaseRepository):
    """Repository for skeleton document database operations."""
    
    def save_skeleton_document(self, skeleton_doc: SkeletonDocument) -> bool:
        """Save skeleton document to database."""
        query = """
        INSERT INTO skeleton_documents (
            id, source_document_ids, template_path, content_blocks,
            delimiter_positions, total_paragraphs_processed, total_clusters_found,
            blocks_generated, uncertain_blocks
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            source_document_ids = EXCLUDED.source_document_ids,
            template_path = EXCLUDED.template_path,
            content_blocks = EXCLUDED.content_blocks,
            delimiter_positions = EXCLUDED.delimiter_positions,
            total_paragraphs_processed = EXCLUDED.total_paragraphs_processed,
            total_clusters_found = EXCLUDED.total_clusters_found,
            blocks_generated = EXCLUDED.blocks_generated,
            uncertain_blocks = EXCLUDED.uncertain_blocks;
        """
        
        try:
            # Format content blocks and delimiter positions as JSONB
            content_blocks_json = self.format_json_for_db(skeleton_doc.content_blocks)
            delimiter_positions_json = self.format_json_for_db(skeleton_doc.delimiter_positions)
            
            params = (
                skeleton_doc.id,
                skeleton_doc.source_document_ids,
                skeleton_doc.template_path,
                content_blocks_json,
                delimiter_positions_json,
                skeleton_doc.total_paragraphs_processed,
                skeleton_doc.total_clusters_found,
                skeleton_doc.blocks_generated,
                skeleton_doc.uncertain_blocks
            )
            
            self.execute_query(query, params, fetch=False)
            logger.info(f"Saved skeleton document {skeleton_doc.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save skeleton document {skeleton_doc.id}: {e}")
            return False
    
    def get_skeleton_document_by_id(self, skeleton_id: str) -> Optional[SkeletonDocument]:
        """Get skeleton document by ID."""
        query = "SELECT * FROM skeleton_documents WHERE id = %s;"
        
        try:
            results = self.execute_query(query, (skeleton_id,))
            if results:
                return self._row_to_skeleton_document(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get skeleton document {skeleton_id}: {e}")
            return None
    
    def get_all_skeleton_documents(self) -> List[SkeletonDocument]:
        """Get all skeleton documents."""
        query = "SELECT * FROM skeleton_documents ORDER BY created_at DESC;"
        
        try:
            results = self.execute_query(query)
            return [self._row_to_skeleton_document(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get all skeleton documents: {e}")
            return []
    
    def get_skeleton_documents_by_source_document(self, source_document_id: str) -> List[SkeletonDocument]:
        """Get skeleton documents that used a specific source document."""
        query = "SELECT * FROM skeleton_documents WHERE %s = ANY(source_document_ids) ORDER BY created_at DESC;"
        
        try:
            results = self.execute_query(query, (source_document_id,))
            return [self._row_to_skeleton_document(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get skeleton documents by source document {source_document_id}: {e}")
            return []
    
    def get_recent_skeleton_documents(self, limit: int = 10) -> List[SkeletonDocument]:
        """Get most recently created skeleton documents."""
        query = "SELECT * FROM skeleton_documents ORDER BY created_at DESC LIMIT %s;"
        
        try:
            results = self.execute_query(query, (limit,))
            return [self._row_to_skeleton_document(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get recent skeleton documents: {e}")
            return []
    
    def update_skeleton_document_template_path(self, skeleton_id: str, new_template_path: str) -> bool:
        """Update skeleton document template path."""
        query = "UPDATE skeleton_documents SET template_path = %s WHERE id = %s;"
        
        try:
            self.execute_query(query, (new_template_path, skeleton_id), fetch=False)
            logger.info(f"Updated skeleton document {skeleton_id} template path")
            return True
        except Exception as e:
            logger.error(f"Failed to update skeleton document {skeleton_id} template path: {e}")
            return False
    
    def update_skeleton_document_statistics(self, skeleton_id: str, 
                                          total_paragraphs: int = None,
                                          total_clusters: int = None,
                                          blocks_generated: int = None,
                                          uncertain_blocks: int = None) -> bool:
        """Update skeleton document processing statistics."""
        updates = []
        params = []
        
        if total_paragraphs is not None:
            updates.append("total_paragraphs_processed = %s")
            params.append(total_paragraphs)
        
        if total_clusters is not None:
            updates.append("total_clusters_found = %s")
            params.append(total_clusters)
        
        if blocks_generated is not None:
            updates.append("blocks_generated = %s")
            params.append(blocks_generated)
        
        if uncertain_blocks is not None:
            updates.append("uncertain_blocks = %s")
            params.append(uncertain_blocks)
        
        if not updates:
            return True  # Nothing to update
        
        params.append(skeleton_id)
        query = f"UPDATE skeleton_documents SET {', '.join(updates)} WHERE id = %s;"
        
        try:
            self.execute_query(query, tuple(params), fetch=False)
            logger.debug(f"Updated skeleton document {skeleton_id} statistics")
            return True
        except Exception as e:
            logger.error(f"Failed to update skeleton document {skeleton_id} statistics: {e}")
            return False
    
    def delete_skeleton_document(self, skeleton_id: str) -> bool:
        """Delete skeleton document."""
        query = "DELETE FROM skeleton_documents WHERE id = %s;"
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (skeleton_id,))
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Deleted skeleton document {skeleton_id}")
                    return True
                else:
                    logger.warning(f"Skeleton document {skeleton_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete skeleton document {skeleton_id}: {e}")
            return False
    
    def search_skeleton_documents_by_template_path(self, path_pattern: str) -> List[SkeletonDocument]:
        """Search skeleton documents by template path pattern."""
        query = "SELECT * FROM skeleton_documents WHERE template_path ILIKE %s ORDER BY created_at DESC;"
        
        try:
            pattern = f"%{path_pattern}%"
            results = self.execute_query(query, (pattern,))
            return [self._row_to_skeleton_document(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to search skeleton documents by template path '{path_pattern}': {e}")
            return []
    
    def get_skeleton_documents_created_after(self, after_date: str) -> List[SkeletonDocument]:
        """Get skeleton documents created after a specific date."""
        query = "SELECT * FROM skeleton_documents WHERE created_at > %s ORDER BY created_at DESC;"
        
        try:
            results = self.execute_query(query, (after_date,))
            return [self._row_to_skeleton_document(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get skeleton documents created after {after_date}: {e}")
            return []
    
    def get_skeleton_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about skeleton documents in the database."""
        try:
            with self.get_cursor(commit=False) as cursor:
                # Total skeleton documents
                cursor.execute("SELECT COUNT(*) as total FROM skeleton_documents;")
                total_skeletons = cursor.fetchone()['total']
                
                # Processing statistics
                cursor.execute("""
                    SELECT 
                        AVG(total_paragraphs_processed) as avg_paragraphs,
                        AVG(total_clusters_found) as avg_clusters,
                        AVG(blocks_generated) as avg_blocks,
                        AVG(uncertain_blocks) as avg_uncertain_blocks,
                        SUM(total_paragraphs_processed) as total_paragraphs_processed,
                        SUM(total_clusters_found) as total_clusters_processed
                    FROM skeleton_documents;
                """)
                processing_stats = cursor.fetchone()
                
                # Template file statistics
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT template_path) as unique_templates,
                        COUNT(*) as total_entries
                    FROM skeleton_documents
                    WHERE template_path IS NOT NULL;
                """)
                template_stats = cursor.fetchone()
                
                # Block generation efficiency
                cursor.execute("""
                    SELECT 
                        SUM(blocks_generated) as total_blocks_generated,
                        SUM(uncertain_blocks) as total_uncertain_blocks
                    FROM skeleton_documents;
                """)
                block_stats = cursor.fetchone()
                
                # Most productive skeleton documents
                cursor.execute("""
                    SELECT id, total_paragraphs_processed, total_clusters_found, blocks_generated
                    FROM skeleton_documents
                    ORDER BY total_paragraphs_processed DESC
                    LIMIT 5;
                """)
                top_skeletons = [dict(row) for row in cursor.fetchall()]
                
                total_blocks = block_stats['total_blocks_generated'] or 0
                total_uncertain = block_stats['total_uncertain_blocks'] or 0
                
                return {
                    "total_skeleton_documents": total_skeletons,
                    "processing_averages": {
                        "paragraphs_per_skeleton": round(float(processing_stats['avg_paragraphs']), 2) if processing_stats['avg_paragraphs'] else 0,
                        "clusters_per_skeleton": round(float(processing_stats['avg_clusters']), 2) if processing_stats['avg_clusters'] else 0,
                        "blocks_per_skeleton": round(float(processing_stats['avg_blocks']), 2) if processing_stats['avg_blocks'] else 0,
                        "uncertain_blocks_per_skeleton": round(float(processing_stats['avg_uncertain_blocks']), 2) if processing_stats['avg_uncertain_blocks'] else 0
                    },
                    "processing_totals": {
                        "total_paragraphs_processed": processing_stats['total_paragraphs_processed'] or 0,
                        "total_clusters_processed": processing_stats['total_clusters_processed'] or 0,
                        "total_blocks_generated": total_blocks,
                        "total_uncertain_blocks": total_uncertain
                    },
                    "template_info": {
                        "unique_template_files": template_stats['unique_templates'] or 0,
                        "total_template_references": template_stats['total_entries'] or 0
                    },
                    "block_efficiency": {
                        "certain_block_percentage": round(((total_blocks - total_uncertain) / total_blocks * 100), 2) if total_blocks > 0 else 0,
                        "uncertain_block_percentage": round((total_uncertain / total_blocks * 100), 2) if total_blocks > 0 else 0
                    },
                    "top_skeleton_documents": top_skeletons
                }
                
        except Exception as e:
            logger.error(f"Failed to get skeleton document statistics: {e}")
            return {}
    
    def get_skeleton_documents_with_content_analysis(self) -> List[Dict[str, Any]]:
        """Get skeleton documents with content block analysis."""
        query = "SELECT * FROM skeleton_documents ORDER BY created_at DESC;"
        
        try:
            results = self.execute_query(query)
            analyzed_skeletons = []
            
            for row in results:
                skeleton_doc = self._row_to_skeleton_document(row)
                
                # Analyze content blocks
                content_analysis = {
                    "total_content_blocks": len(skeleton_doc.content_blocks),
                    "blocks_with_alternatives": sum(1 for block_list in skeleton_doc.content_blocks.values() if len(block_list) > 1),
                    "total_paragraph_alternatives": sum(len(block_list) for block_list in skeleton_doc.content_blocks.values()),
                    "average_alternatives_per_block": 0
                }
                
                if content_analysis["total_content_blocks"] > 0:
                    content_analysis["average_alternatives_per_block"] = round(
                        content_analysis["total_paragraph_alternatives"] / content_analysis["total_content_blocks"], 2
                    )
                
                analyzed_skeletons.append({
                    "skeleton_document": skeleton_doc,
                    "content_analysis": content_analysis
                })
            
            return analyzed_skeletons
        except Exception as e:
            logger.error(f"Failed to get skeleton documents with content analysis: {e}")
            return []
    
    def _row_to_skeleton_document(self, row: Dict[str, Any]) -> SkeletonDocument:
        """Convert database row to SkeletonDocument object."""
        # Parse JSONB fields
        content_blocks = self.parse_json_from_db(row['content_blocks']) if row['content_blocks'] else {}
        delimiter_positions = self.parse_json_from_db(row['delimiter_positions']) if row['delimiter_positions'] else {}
        
        # Convert content_blocks keys to integers if they're strings
        if content_blocks:
            content_blocks = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in content_blocks.items()}
        
        # Convert delimiter_positions keys to integers if they're strings  
        if delimiter_positions:
            delimiter_positions = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in delimiter_positions.items()}
        
        return SkeletonDocument(
            id=row['id'],
            source_document_ids=row['source_document_ids'] or [],
            template_path=row['template_path'] or "",
            content_blocks=content_blocks,
            delimiter_positions=delimiter_positions,
            total_paragraphs_processed=row['total_paragraphs_processed'] or 0,
            total_clusters_found=row['total_clusters_found'] or 0,
            blocks_generated=row.get('blocks_generated', 0) or 0,
            uncertain_blocks=row.get('uncertain_blocks', 0) or 0
        )
