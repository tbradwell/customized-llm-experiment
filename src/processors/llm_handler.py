"""Centralized LLM interaction logic for contract generation."""

import logging
from typing import Dict, Any

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMHandler:
    """Centralized LLM interaction logic."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        # self.temperature = settings.openai_temperature  # Use default temperature to avoid API errors
    
    def process_with_llm(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Process a prompt with the LLM."""
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert legal document completion specialist. Be precise and factual."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            
            response = self.client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")            # Only add temperature if not using default value
            raise
    
    def create_prompt(self, prompt_type: str, **kwargs) -> str:
        """Create prompts for different types of processing."""
        
        if prompt_type == "deletable_paragraph":
            return self._create_deletable_paragraph_prompt(**kwargs)
        elif prompt_type == "fillable_paragraph":
            return self._create_fillable_paragraph_prompt(**kwargs)
        elif prompt_type == "template_block":
            return self._create_template_block_prompt(**kwargs)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    def _create_deletable_paragraph_prompt(self, paragraph_text: str, new_data: str) -> str:
        """Create prompt for deletable paragraph processing."""
        return f"""
You are a legal document editor. The paragraph below contains {{~~}} placeholders that can be deleted if no suitable information exists in the new data.

PARAGRAPH TO PROCESS:
{paragraph_text}

NEW DATA AVAILABLE:
{new_data}

INSTRUCTIONS:
1. If the NEW DATA contains relevant information for the {{~~}} placeholder, fill it with specific details
2. If the NEW DATA does not contain relevant information, DELETE the entire paragraph by returning empty string
3. Do not use generic text like "יש לציין" or "יינתן במועד מאוחר יותר"
4. Use only specific, factual information from the NEW DATA

Return the processed paragraph (or empty string to delete):
"""
    
    def _create_fillable_paragraph_prompt(self, paragraph_text: str, new_data: str) -> str:
        """Create prompt for fillable paragraph processing."""
        return f"""
You are a legal document editor. Fill the placeholders in the paragraph below using specific information from the new data.

PARAGRAPH TO PROCESS:
{paragraph_text}

NEW DATA AVAILABLE:
{new_data}

INSTRUCTIONS:
1. Replace ALL placeholders {{}} with specific information from the NEW DATA
2. NEVER use generic placeholders like [השלם], [כתובת], [שם]
3. If specific information is not available and the place holder contains {{~~}} you can return empty text, otherwise use reasonable legal text based on context
4. If the NEW DATA is given in Hebrew, then the output should be in Hebrew. 

Return the completed paragraph with ALL placeholders filled:
"""
    
    def _create_template_block_prompt(self, paragraphs_text: str, new_data: str, 
                                    par_separator: str, block_separator: str) -> str:
        """Create prompt for template block processing."""
        return f"""
You are a legal document editor. You are given multiple paragraphs separated by "{par_separator}" that contain placeholders to be filled.

PARAGRAPHS TO PROCESS (separated by {par_separator}):
{paragraphs_text}

NEW DATA AVAILABLE:
{new_data}

INSTRUCTIONS:
1. Fill ALL placeholders {{}} in each paragraph using specific information from the NEW DATA
2. Keep the separation markers "{par_separator}" between paragraphs in your response
3. NEVER use generic placeholders like [השלם], [כתובת], [שם]
4. Use only specific, factual information from the NEW DATA
5. If the NEW DATA is given in Hebrew, then the output should be in Hebrew. 
6. If there are more than one posability to fill the paragraph (like multiple plaintiffs), duplicate your answer and seperate it with {block_separator}

Return the filled paragraphs with separation markers:
"""
