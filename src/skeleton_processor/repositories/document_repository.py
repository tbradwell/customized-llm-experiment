"""Document repository for database operations."""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """Repository for document database operations."""
    
    def save_document(self, name: str, path: str) -> str:
        """Save document to database and return document ID."""
        document_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO documents (id, name, path, created_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (id) DO NOTHING;
        """
        
        try:
            self.execute_query(query, (document_id, name, path), fetch=False)
            logger.info(f"Saved document {name} with ID {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Failed to save document {name}: {e}")
            raise
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        query = "SELECT * FROM documents WHERE id = %s;"
        
        try:
            results = self.execute_query(query, (document_id,))
            return dict(results[0]) if results else None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def get_document_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Get document by file path."""
        query = "SELECT * FROM documents WHERE path = %s;"
        
        try:
            results = self.execute_query(query, (path,))
            return dict(results[0]) if results else None
        except Exception as e:
            logger.error(f"Failed to get document by path {path}: {e}")
            return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        query = "SELECT * FROM documents ORDER BY created_at DESC;"
        
        try:
            results = self.execute_query(query)
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []
    
    def get_documents_by_ids(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs."""
        if not document_ids:
            return []
        
        placeholders = ','.join(['%s'] * len(document_ids))
        query = f"SELECT * FROM documents WHERE id IN ({placeholders}) ORDER BY created_at;"
        
        try:
            results = self.execute_query(query, tuple(document_ids))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return []
    
    def update_document_path(self, document_id: str, new_path: str) -> bool:
        """Update document path."""
        query = "UPDATE documents SET path = %s WHERE id = %s;"
        
        try:
            self.execute_query(query, (new_path, document_id), fetch=False)
            logger.info(f"Updated document {document_id} path to {new_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document {document_id} path: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all related paragraphs."""
        try:
            with self.get_cursor() as cursor:
                # Delete related paragraphs first
                cursor.execute("DELETE FROM paragraphs WHERE document_id = %s;", (document_id,))
                deleted_paragraphs = cursor.rowcount
                
                # Delete document
                cursor.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
                deleted_documents = cursor.rowcount
                
                if deleted_documents > 0:
                    logger.info(f"Deleted document {document_id} and {deleted_paragraphs} related paragraphs")
                    return True
                else:
                    logger.warning(f"Document {document_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about documents in the database."""
        try:
            with self.get_cursor(commit=False) as cursor:
                # Total documents
                cursor.execute("SELECT COUNT(*) as total FROM documents;")
                total_docs = cursor.fetchone()['total']
                
                # Total paragraphs across all documents
                cursor.execute("SELECT COUNT(*) as total FROM paragraphs;")
                total_paragraphs = cursor.fetchone()['total']
                
                # Average paragraphs per document
                cursor.execute("""
                    SELECT AVG(paragraph_count) as avg_paragraphs 
                    FROM (
                        SELECT document_id, COUNT(*) as paragraph_count 
                        FROM paragraphs 
                        GROUP BY document_id
                    ) doc_counts;
                """)
                avg_paragraphs_result = cursor.fetchone()
                avg_paragraphs = float(avg_paragraphs_result['avg_paragraphs']) if avg_paragraphs_result['avg_paragraphs'] else 0.0
                
                # Documents with most/least paragraphs
                cursor.execute("""
                    SELECT d.name, COUNT(p.id) as paragraph_count
                    FROM documents d
                    LEFT JOIN paragraphs p ON d.id = p.document_id
                    GROUP BY d.id, d.name
                    ORDER BY paragraph_count DESC
                    LIMIT 5;
                """)
                top_documents = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "total_documents": total_docs,
                    "total_paragraphs": total_paragraphs,
                    "average_paragraphs_per_document": round(avg_paragraphs, 2),
                    "top_documents_by_paragraph_count": top_documents
                }
                
        except Exception as e:
            logger.error(f"Failed to get document statistics: {e}")
            return {
                "total_documents": 0,
                "total_paragraphs": 0,
                "average_paragraphs_per_document": 0.0,
                "top_documents_by_paragraph_count": []
            }
    
    def search_documents_by_name(self, name_pattern: str) -> List[Dict[str, Any]]:
        """Search documents by name pattern."""
        query = "SELECT * FROM documents WHERE name ILIKE %s ORDER BY created_at DESC;"
        
        try:
            # Add wildcards for pattern matching
            pattern = f"%{name_pattern}%"
            results = self.execute_query(query, (pattern,))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to search documents by name '{name_pattern}': {e}")
            return []
    
    def get_documents_created_after(self, after_date: str) -> List[Dict[str, Any]]:
        """Get documents created after a specific date."""
        query = "SELECT * FROM documents WHERE created_at > %s ORDER BY created_at DESC;"
        
        try:
            results = self.execute_query(query, (after_date,))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get documents created after {after_date}: {e}")
            return []
