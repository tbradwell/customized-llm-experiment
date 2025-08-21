"""Paragraph processing utilities for contract generation."""

import logging
import re
from typing import Tuple

from .llm_handler import LLMHandler
from ..utils.doc_handler import DocHandler

logger = logging.getLogger(__name__)


class ParagraphProcessor:
    """Handles individual paragraph processing."""
    
    def __init__(self):
        self.llm_handler = LLMHandler()
    
    def fill_paragraph_with_data(self, paragraph, new_data: str) -> bool:
        """Fill a single paragraph with new data."""
        logger.info(f"Processing paragraph: {paragraph.text[:100]}...")
        
        new_text, paragraph_changed = self._process_paragraph(paragraph.text, new_data)
        
        if paragraph_changed:
            if DocHandler.replace_paragraph_text(paragraph, new_text):
                logger.info(f"Updated paragraph to: {new_text[:100]}...")
                return True
            else:
                logger.warning("Failed to replace paragraph text")
        
        return False
    
    def _process_paragraph(self, paragraph_text: str, new_data: str) -> Tuple[str, bool]:
        """Process a single paragraph containing placeholders."""
        
        # Check for {~~} placeholders - these can be deleted if no suitable information
        if '{~' in paragraph_text and '~}' in paragraph_text:
            return self._process_deletable_paragraph(paragraph_text, new_data)
        
        # Check for regular placeholders
        elif '{' in paragraph_text and '}' in paragraph_text:
            return self._process_fillable_paragraph(paragraph_text, new_data)
        
        return paragraph_text, False
    
    def _process_deletable_paragraph(self, paragraph_text: str, new_data: str) -> Tuple[str, bool]:
        """Process paragraph with {~~} placeholders that can be deleted."""
        
        prompt = self.llm_handler.create_prompt(
            "deletable_paragraph",
            paragraph_text=paragraph_text,
            new_data=new_data[:1500]
        )
        
        try:
            new_text = self.llm_handler.process_with_llm(prompt)
            
            # Check if significantly different
            if new_text != paragraph_text:
                return new_text, True
            
            return paragraph_text, False
            
        except Exception as e:
            logger.error(f"Failed to process deletable paragraph: {e}")
            return paragraph_text, False
    
    def _process_fillable_paragraph(self, paragraph_text: str, new_data: str) -> Tuple[str, bool]:
        """Process paragraph with regular {} placeholders."""
        
        prompt = self.llm_handler.create_prompt(
            "fillable_paragraph",
            paragraph_text=paragraph_text,
            new_data=new_data
        )
        
        try:
            new_text = self.llm_handler.process_with_llm(prompt)
            
            # Check if placeholders were actually filled
            remaining_placeholders = re.findall(r'\{[^}]*\}', new_text)
            original_placeholders = re.findall(r'\{[^}]*\}', paragraph_text)
            
            if len(remaining_placeholders) < len(original_placeholders):
                return new_text, True
            
            return paragraph_text, False
            
        except Exception as e:
            logger.error(f"Failed to process fillable paragraph: {e}")
            return paragraph_text, False
