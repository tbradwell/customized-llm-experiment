"""Template block processing utilities for contract generation."""

import logging
import re
from typing import List, Tuple

from .llm_handler import LLMHandler

logger = logging.getLogger(__name__)


class TemplateProcessor:
    """Handles template blocks with {% %} syntax."""
    
    def __init__(self):
        self.llm_handler = LLMHandler()
    
    def fill_template_block(self, paragraphs: List[str], new_data: str) -> Tuple[List[List[str]], bool]:
        """Fill template block with {% %} placeholders that contain multiple paragraphs."""
        
        # Process the paragraphs iteratively with the LLM
        filled_content = self._fill_template_block(paragraphs, new_data)
        
        return filled_content, True if len(filled_content) > 0 else False
    
    def _fill_template_block(self, paragraphs: List[str], new_data: str) -> List[List[str]]:
        """Fill template paragraphs iteratively until LLM says no more to fill."""
        
        # Join paragraphs with separation markers
        separation_marker_par = "---PARAGRAPH_SEPARATOR---"
        separation_marker_block = "---BLOCK---"
        
        combined_paragraphs = f"\n{separation_marker_par}\n".join(paragraphs)
        
        # Send to LLM for filling
        new_content = self._send_paragraphs_to_llm(
            combined_paragraphs, new_data, separation_marker_par, separation_marker_block
        )
        
        # Process the result
        result = self._process_llm_response(new_content, separation_marker_par, separation_marker_block)
        
        return result
    
    def _send_paragraphs_to_llm(self, paragraphs_text: str, new_data: str, 
                                par_separator: str, block_separator: str) -> str:
        """Send paragraphs to LLM for filling with proper separation handling."""
        
        prompt = self.llm_handler.create_prompt(
            "template_block",
            paragraphs_text=paragraphs_text,
            new_data=new_data,
            par_separator=par_separator,
            block_separator=block_separator
        )
        
        try:
            return self.llm_handler.process_with_llm(prompt)
        except Exception as e:
            logger.error(f"Failed to process template paragraphs: {e}")
            return ""
    
    def _process_llm_response(self, new_content: str, par_separator: str, block_separator: str) -> List[List[str]]:
        """Process LLM response and clean up separation markers."""
        
        # Remove separation markers from final result
        result = new_content.replace(f"\n{par_separator}\n", "\n").replace(f"{par_separator}", "\n")
        result_blocks = result.split(block_separator)
        
        clean_block = []
        for block in result_blocks:
            clean_paragraphs = []
            paragraphs = block.split('\n')
            for paragraph in paragraphs:
                new_paragraph = paragraph.replace('%}', '').replace('{%', '')
                if new_paragraph != '':
                    clean_paragraphs.append(new_paragraph)
            if len(clean_paragraphs) > 0:
                clean_block.append(clean_paragraphs)
        
        return clean_block
    
    def extract_template_blocks(self, paragraph_list: List[str]) -> List[Tuple[str, str]]:
        """Extract and clean paragraphs from template block list."""
        cleaned_paragraphs = []
        original_content = "\n".join(paragraph_list)
        
        for paragraph in paragraph_list:
            # Clean template syntax and whitespace
            cleaned = paragraph.strip()
            
            # Remove {% template syntax
            cleaned = re.sub(r'\{\%.*?\%\}', '', cleaned)
            
            # Remove %} endings
            cleaned = re.sub(r'\%\}', '', cleaned)
            
            # Clean excessive whitespace and tabs
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Only add non-empty paragraphs
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        
        # Return as expected tuple format
        if cleaned_paragraphs:
            cleaned_content = "\n".join(cleaned_paragraphs)
            return [(cleaned_content, original_content)]
        
        return []
