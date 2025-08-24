"""Text cleaning utilities for skeleton processor."""

import re
import logging
from typing import Dict, List, Tuple, Set
import spacy
from spacy import displacy

logger = logging.getLogger(__name__)


class TextCleaner:
    """Utility class for cleaning text as per algorithm steps 1.2.1-1.2.2."""
    
    def __init__(self, language: str = "he"):
        """Initialize text cleaner.
        
        Args:
            language: Language model to use for NER (he for Hebrew, en for English)
        """
        self.language = language
        self.nlp = None
        self._load_language_model()
        
        # Placeholders
        self.entity_placeholder = "[ENTITY]"
        self.number_placeholder = "[NUMBER]"
        
        # Patterns for number detection
        self.number_patterns = [
            r'\b\d+\.?\d*\b',  # Basic numbers (123, 123.45)
            r'\b\d{1,3}(?:,\d{3})*\b',  # Numbers with commas (1,000)
            r'\b\d+%\b',  # Percentages (50%)
            r'\b\d+/\d+\b',  # Fractions (1/2)
            r'₪\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # Israeli Shekel amounts
            r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
            r'€\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # Euro amounts
        ]
        
        # Hebrew number words (for Hebrew text)
        self.hebrew_numbers = {
            'אחד', 'שנים', 'שלושה', 'ארבעה', 'חמישה', 'שישה', 'שבעה', 'שמונה', 
            'תשעה', 'עשרה', 'עשרים', 'שלושים', 'ארבעים', 'חמישים', 'שישים',
            'שבעים', 'שמונים', 'תשעים', 'מאה', 'אלף', 'מיליון', 'מיליארד',
            'ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שביעי', 'שמיני',
            'תשיעי', 'עשירי'
        }
    
    def _load_language_model(self):
        """Load spaCy language model for NER."""
        try:
            if self.language == "he":
                # Try to load Hebrew model, fallback to English if not available
                try:
                    self.nlp = spacy.load("he_core_news_sm")
                except OSError:
                    logger.warning("Hebrew spaCy model not found, using English model")
                    self.nlp = spacy.load("en_core_web_sm")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                
            logger.info(f"Loaded spaCy model: {self.nlp.meta['name']}")
            
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def clean_text(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """Clean text by replacing named entities and numbers with placeholders.
        
        Args:
            text: Original text to clean
            
        Returns:
            Tuple of (cleaned_text, replacements_map)
        """
        if not text.strip():
            return text, {}
        
        cleaned_text = text
        replacements = {
            'entities': [],
            'numbers': []
        }
        
        # Step 1.2.1: Remove named entities
        cleaned_text, entities = self._replace_named_entities(cleaned_text)
        replacements['entities'] = entities
        
        # Step 1.2.2: Remove numbers
        cleaned_text, numbers = self._replace_numbers(cleaned_text)
        replacements['numbers'] = numbers
        
        logger.debug(f"Cleaned text: {len(entities)} entities, {len(numbers)} numbers replaced")
        
        return cleaned_text, replacements
    
    def _replace_named_entities(self, text: str) -> Tuple[str, List[str]]:
        """Replace named entities with placeholders.
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (cleaned_text, list_of_replaced_entities)
        """
        if not self.nlp:
            logger.warning("No spaCy model loaded, skipping NER")
            return text, []
        
        try:
            doc = self.nlp(text)
            entities = []
            cleaned_text = text
            
            # Sort entities by position (reverse order to maintain indices)
            sorted_entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)
            
            for ent in sorted_entities:
                # Skip certain entity types that might be structural
                if ent.label_ in ['CARDINAL', 'ORDINAL', 'QUANTITY', 'PERCENT', 'MONEY']:
                    continue
                
                entities.append(ent.text)
                
                # Replace entity with placeholder
                cleaned_text = (
                    cleaned_text[:ent.start_char] + 
                    self.entity_placeholder + 
                    cleaned_text[ent.end_char:]
                )
            
            return cleaned_text, list(reversed(entities))  # Reverse to maintain original order
            
        except Exception as e:
            logger.error(f"Error in named entity replacement: {e}")
            return text, []
    
    def _replace_numbers(self, text: str) -> Tuple[str, List[str]]:
        """Replace numbers with placeholders.
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (cleaned_text, list_of_replaced_numbers)
        """
        numbers = []
        cleaned_text = text
        
        # Replace Hebrew number words first (if Hebrew language)
        if self.language == "he":
            cleaned_text, hebrew_nums = self._replace_hebrew_numbers(cleaned_text)
            numbers.extend(hebrew_nums)
        
        # Replace numeric patterns
        for pattern in self.number_patterns:
            matches = list(re.finditer(pattern, cleaned_text))
            
            # Process matches in reverse order to maintain indices
            for match in reversed(matches):
                numbers.append(match.group())
                cleaned_text = (
                    cleaned_text[:match.start()] + 
                    self.number_placeholder + 
                    cleaned_text[match.end():]
                )
        
        return cleaned_text, list(reversed(numbers))  # Reverse to maintain original order
    
    def _replace_hebrew_numbers(self, text: str) -> Tuple[str, List[str]]:
        """Replace Hebrew number words with placeholders.
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (cleaned_text, list_of_replaced_hebrew_numbers)
        """
        hebrew_nums = []
        cleaned_text = text
        
        # Create pattern for Hebrew numbers
        hebrew_pattern = r'\b(?:' + '|'.join(self.hebrew_numbers) + r')\b'
        
        matches = list(re.finditer(hebrew_pattern, cleaned_text))
        
        # Process matches in reverse order
        for match in reversed(matches):
            hebrew_nums.append(match.group())
            cleaned_text = (
                cleaned_text[:match.start()] + 
                self.number_placeholder + 
                cleaned_text[match.end():]
            )
        
        return cleaned_text, list(reversed(hebrew_nums))
    
    def restore_text(self, cleaned_text: str, replacements: Dict[str, List[str]]) -> str:
        """Restore original text from cleaned version.
        
        Args:
            cleaned_text: Text with placeholders
            replacements: Dictionary with original entities and numbers
            
        Returns:
            Restored original text
        """
        restored_text = cleaned_text
        
        # Restore entities
        entities = replacements.get('entities', [])
        for entity in entities:
            restored_text = restored_text.replace(self.entity_placeholder, entity, 1)
        
        # Restore numbers
        numbers = replacements.get('numbers', [])
        for number in numbers:
            restored_text = restored_text.replace(self.number_placeholder, number, 1)
        
        return restored_text
    
    def get_cleaning_stats(self, text: str) -> Dict[str, int]:
        """Get statistics about what would be cleaned from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with cleaning statistics
        """
        _, replacements = self.clean_text(text)
        
        return {
            'original_length': len(text),
            'entities_found': len(replacements.get('entities', [])),
            'numbers_found': len(replacements.get('numbers', [])),
            'total_replacements': len(replacements.get('entities', [])) + len(replacements.get('numbers', []))
        }
