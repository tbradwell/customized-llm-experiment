"""Document preprocessor for skeleton processor algorithm steps 1.1-1.2.6."""

import logging
import os
from typing import List, Dict, Tuple, Any
import uuid
from docx import Document

from .models.paragraph import Paragraph
from .utils.embedding_client import EmbeddingClient
from .utils.text_cleaner import TextCleaner
from ..utils.doc_handler import DocHandler
from .repositories.document_repository import DocumentRepository
from .repositories.paragraph_repository import ParagraphRepository
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Handles document loading, paragraph extraction, and embedding creation."""
    
    def __init__(self, embedding_client: EmbeddingClient, text_cleaner: TextCleaner,
                 document_repository: Optional[DocumentRepository] = None,
                 paragraph_repository: Optional[ParagraphRepository] = None):
        """Initialize document preprocessor.
        
        Args:
            embedding_client: Client for creating embeddings
            text_cleaner: Client for text cleaning operations
            document_repository: Optional repository for document database operations
            paragraph_repository: Optional repository for paragraph database operations
        """
        self.embedding_client = embedding_client
        self.text_cleaner = text_cleaner
        self.doc_handler = DocHandler()
        self.document_repository = document_repository
        self.paragraph_repository = paragraph_repository
        self.use_database = document_repository is not None and paragraph_repository is not None
        
    def process_documents(self, document_paths: List[str]) -> List[Paragraph]:
        """Process multiple documents according to algorithm steps 1.1-1.2.6.
        
        Args:
            document_paths: List of paths to DOCX documents
            
        Returns:
            List of processed paragraphs with embeddings
        """
        logger.info(f"Processing {len(document_paths)} documents")
        
        all_paragraphs = []
        
        # Step 1: For each document in documents
        for doc_path in document_paths:
            try:
                # Step 1.1: Save document to vector db (document name, document id)
                document_id = self._save_document_to_db(doc_path)
                logger.info(f"Processing document: {doc_path} (ID: {document_id})")
                
                # Step 1.2: For each paragraph in document
                doc_paragraphs = self._process_single_document(doc_path, document_id)
                
                # Save paragraphs to database if available
                if self.use_database and doc_paragraphs:
                    saved_count = self.paragraph_repository.save_paragraphs_batch(doc_paragraphs)
                    logger.info(f"Saved {saved_count} paragraphs to database")
                
                all_paragraphs.extend(doc_paragraphs)
                
                logger.info(f"Processed {len(doc_paragraphs)} paragraphs from {doc_path}")
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_path}: {e}")
                continue
        
        logger.info(f"Total paragraphs processed: {len(all_paragraphs)}")
        return all_paragraphs
    
    def _process_single_document(self, doc_path: str, document_id: str) -> List[Paragraph]:
        """Process a single document and extract paragraphs.
        
        Args:
            doc_path: Path to the document
            document_id: Unique identifier for the document
            
        Returns:
            List of processed paragraphs
        """
        # Load document using doc_handler
        doc = Document(doc_path)
        paragraphs = []
        
        # Extract text and metadata
        doc_paragraphs = [p for p in doc.paragraphs if p.text.strip()]
        total_paragraphs = len(doc_paragraphs)
        
        logger.debug(f"Found {total_paragraphs} non-empty paragraphs in {doc_path}")
        
        # Process each paragraph
        for position, doc_paragraph in enumerate(doc_paragraphs):
            try:
                paragraph = self._process_paragraph(
                    doc_paragraph, position, total_paragraphs, document_id
                )
                if paragraph:
                    paragraphs.append(paragraph)
                    
            except Exception as e:
                logger.warning(f"Failed to process paragraph {position} in {doc_path}: {e}")
                continue
        
        return paragraphs
    
    def _process_paragraph(self, doc_paragraph, position: int, total_paragraphs: int, 
                          document_id: str) -> Paragraph:
        """Process a single paragraph according to algorithm steps 1.2.1-1.2.6.
        
        Args:
            doc_paragraph: python-docx paragraph object
            position: Absolute position in document
            total_paragraphs: Total number of paragraphs in document
            document_id: Document identifier
            
        Returns:
            Processed Paragraph object
        """
        original_text = doc_paragraph.text.strip()
        
        if not original_text:
            return None
        
        # Step 1.2.1: Create an embedding of the paragraph with OpenAI (before cleaning)
        try:
            text_embedding = self.embedding_client.create_embedding(original_text)
        except Exception as e:
            logger.error(f"Failed to create embedding for paragraph {position}: {e}")
            raise
        
        # Step 1.2.2: Remove all named entities and replace with placeholder (after embedding)
        # Step 1.2.3: Remove all numbers and replace with placeholder
        clean_text, replacements = self.text_cleaner.clean_text(original_text)
        
        logger.debug(f"Cleaned paragraph {position}: "
                    f"{len(replacements.get('entities', []))} entities, "
                    f"{len(replacements.get('numbers', []))} numbers replaced")
        
        # Step 1.2.4: Calculate relative position (no longer concatenated to embedding)
        relative_position = position / total_paragraphs if total_paragraphs > 0 else 0.0
        
        # Step 1.2.5: Save font style for each run in the paragraph
        font_styles = self._extract_font_styles(doc_paragraph)
        
        # Create Paragraph object
        paragraph = Paragraph(
            id=str(uuid.uuid4()),
            document_id=document_id,
            original_text=original_text,
            clean_text=clean_text,
            absolute_position=position,
            relative_position=relative_position,
            embedding=text_embedding,  # Use text embedding directly (no position concatenation)
            font_style={
                'runs': font_styles,
                'replacements': replacements,
                'alignment': self._get_paragraph_alignment(doc_paragraph)
            }
        )
        
        logger.debug(f"Created paragraph object {paragraph.id} at position {position}")
        return paragraph
    
    def _extract_font_styles(self, doc_paragraph) -> List[Dict[str, Any]]:
        """Extract font styling information from paragraph runs.
        
        Args:
            doc_paragraph: python-docx paragraph object
            
        Returns:
            List of font style dictionaries for each run
        """
        font_styles = []
        
        for run in doc_paragraph.runs:
            try:
                style = {
                    'text': run.text,
                    'bold': run.bold,
                    'italic': run.italic,
                    'underline': run.underline,
                    'font_name': run.font.name,
                    'font_size': run.font.size.pt if run.font.size else None,
                    'color': str(run.font.color.rgb) if run.font.color.rgb else None
                }
                font_styles.append(style)
                
            except Exception as e:
                logger.warning(f"Failed to extract font style from run: {e}")
                # Add basic style information
                font_styles.append({
                    'text': run.text,
                    'bold': None,
                    'italic': None,
                    'underline': None,
                    'font_name': None,
                    'font_size': None,
                    'color': None
                })
        
        return font_styles
    
    def _get_paragraph_alignment(self, doc_paragraph) -> str:
        """Get paragraph alignment information.
        
        Args:
            doc_paragraph: python-docx paragraph object
            
        Returns:
            String representation of alignment
        """
        try:
            if doc_paragraph.alignment is not None:
                from docx.enum.text import WD_ALIGN_PARAGRAPH
                alignment_map = {
                    WD_ALIGN_PARAGRAPH.LEFT: 'left',
                    WD_ALIGN_PARAGRAPH.CENTER: 'center',
                    WD_ALIGN_PARAGRAPH.RIGHT: 'right',
                    WD_ALIGN_PARAGRAPH.JUSTIFY: 'justify'
                }
                return alignment_map.get(doc_paragraph.alignment, 'left')
            return 'left'
            
        except Exception:
            return 'left'
    
    def _save_document_to_db(self, doc_path: str) -> str:
        """Save document to database and return document ID.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Document identifier
        """
        filename = os.path.basename(doc_path)
        
        if self.use_database:
            # Check if document already exists
            existing_doc = self.document_repository.get_document_by_path(doc_path)
            if existing_doc:
                logger.info(f"Document already exists in database: {existing_doc['id']}")
                return existing_doc['id']
            
            # Save new document
            document_id = self.document_repository.save_document(filename, doc_path)
            logger.info(f"Saved document to database: {document_id}")
            return document_id
        else:
            # Generate unique ID for in-memory processing
            timestamp = str(uuid.uuid4())[:8]
            return f"{filename}_{timestamp}"
            
    def _generate_document_id(self, doc_path: str) -> str:
        """Generate a unique document identifier (legacy method).
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Unique document identifier
        """
        return self._save_document_to_db(doc_path)
    
    def process_additional_document(self, doc_path: str, existing_paragraphs: List[Paragraph]) -> List[Paragraph]:
        """Process an additional document for incremental processing.
        
        Args:
            doc_path: Path to the new document
            existing_paragraphs: Previously processed paragraphs
            
        Returns:
            List of newly processed paragraphs
        """
        logger.info(f"Processing additional document: {doc_path}")
        
        document_id = self._generate_document_id(doc_path)
        new_paragraphs = self._process_single_document(doc_path, document_id)
        
        logger.info(f"Processed {len(new_paragraphs)} new paragraphs")
        return new_paragraphs
    
    def validate_paragraphs(self, paragraphs: List[Paragraph]) -> Tuple[bool, List[str]]:
        """Validate processed paragraphs for completeness.
        
        Args:
            paragraphs: List of paragraphs to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not paragraphs:
            issues.append("No paragraphs processed")
            return False, issues
        
        # Check for required fields
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.id:
                issues.append(f"Paragraph {i} missing ID")
            
            if not paragraph.original_text:
                issues.append(f"Paragraph {i} missing original text")
            
            if not paragraph.has_embedding:
                issues.append(f"Paragraph {i} missing embedding")
            
            if not paragraph.document_id:
                issues.append(f"Paragraph {i} missing document ID")
        
        # Validate embeddings
        embeddings = [p.embedding for p in paragraphs if p.has_embedding]
        if embeddings and not self.embedding_client.validate_embeddings(embeddings):
            issues.append("Some embeddings have incorrect dimensions")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"All {len(paragraphs)} paragraphs validated successfully")
        else:
            logger.warning(f"Validation failed: {issues}")
        
        return is_valid, issues
    
    def get_processing_stats(self, paragraphs: List[Paragraph]) -> Dict[str, Any]:
        """Get statistics about processed paragraphs.
        
        Args:
            paragraphs: List of processed paragraphs
            
        Returns:
            Dictionary with processing statistics
        """
        if not paragraphs:
            return {}
        
        # Document statistics
        documents = set(p.document_id for p in paragraphs)
        
        # Text statistics
        total_chars = sum(len(p.original_text) for p in paragraphs)
        avg_chars = total_chars / len(paragraphs)
        
        # Cleaning statistics
        total_entities = sum(
            len(p.font_style.get('replacements', {}).get('entities', []))
            for p in paragraphs
        )
        total_numbers = sum(
            len(p.font_style.get('replacements', {}).get('numbers', []))
            for p in paragraphs
        )
        
        return {
            'total_paragraphs': len(paragraphs),
            'total_documents': len(documents),
            'avg_paragraphs_per_doc': len(paragraphs) / len(documents),
            'total_characters': total_chars,
            'avg_characters_per_paragraph': avg_chars,
            'total_entities_replaced': total_entities,
            'total_numbers_replaced': total_numbers,
            'embedding_dimension': self.embedding_client.get_embedding_dimension()
        }
