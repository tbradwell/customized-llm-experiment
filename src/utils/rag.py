#!/usr/bin/env python3
"""
FODT Text Matcher - Finds the best-suited text from new_data based on FODT content
Uses OpenAI's embedding models for better semantic understanding
"""

from typing import List, Dict, Tuple, Any
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pathlib import Path
import tiktoken

from src.processors.data_loader import DataLoader


class FODTTextMatcher:
    def __init__(self, openai_api_key: str, model_name: str = 'text-embedding-3-small'):
        """
        Initialize with OpenAI embedding model
        
        Args:
            openai_api_key: Your OpenAI API key
            model_name: 'text-embedding-3-small' (faster) or 'text-embedding-3-large' (more accurate)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
 
 
         
    def get_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Get embeddings from OpenAI API with batching and token limit checking"""
        embeddings = []
        
        # Filter texts that are too long (max ~8000 tokens to be safe)
        valid_texts = []
        for text in texts:
            token_count = self.count_tokens(text)
            if token_count > 8000:
                # Truncate text if too long
                tokens = self.tokenizer.encode(text)[:7500]  # Keep some buffer
                truncated_text = self.tokenizer.decode(tokens)
                valid_texts.append(truncated_text)
                print(f"Warning: Truncated text from {token_count} to ~7500 tokens")
            else:
                valid_texts.append(text)
        
        # Process in batches
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                for item in response.data:
                    embeddings.append(item.embedding)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add zero embeddings for failed batch
                embedding_dim = 1536 if 'small' in self.model_name else 3072
                for _ in batch:
                    embeddings.append([0.0] * embedding_dim)
        
        return np.array(embeddings)
        


       
    def chunk_new_data(self, new_data: str, max_tokens: int = 1000) -> List[str]:
        """Split new_data into manageable chunks based on token count"""
        
        # Handle both string and list inputs
        if isinstance(new_data, list):
            # If new_data is a list, join it into a single string
            new_data = "\n\n".join(str(item) for item in new_data)
        elif not isinstance(new_data, str):
            # Convert other types to string
            new_data = str(new_data)
        
        # Split by paragraphs first
        paragraphs = new_data.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current paragraph
                if self.count_tokens(paragraph) <= max_tokens:
                    current_chunk = paragraph
                else:
                    # Split very long paragraphs by sentences
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        test_sentence_chunk = temp_chunk + ". " + sentence if temp_chunk else sentence
                        
                        if self.count_tokens(test_sentence_chunk) <= max_tokens:
                            temp_chunk = test_sentence_chunk
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    
                    current_chunk = temp_chunk
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        print(f"Created {len(chunks)} chunks from new_data")
        return chunks

    def _load_new_data(self, new_data_dir: str) -> str:
        """Load and combine new data from directory."""
        
        data_loader = DataLoader()
        return data_loader.load_new_data(new_data_dir, is_flat_text=False)

    def extract_key_concepts(self, fodt_text: str) -> List[str]:
        """Extract key concepts from FODT text to improve matching"""
        
        # Common legal/document concepts to look for
        patterns = {
            'names': r'[A-Z][a-z]+ [A-Z][a-z]+',  # Person names
            'dates': r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}',  # Dates
            'amounts': r'[\₪$€£]\s?\d+[,\d]*',  # Currency amounts
            'case_numbers': r'[A-Z]{1,3}\s?\d+[-/]\d+',  # Case numbers
            'addresses': r'\d+\s+[A-Za-z\s]+(?:Street|St|Road|Rd|Avenue|Ave)',  # Addresses
            'phone_numbers': r'0\d{1,2}[-\s]?\d{7,8}',  # Phone numbers
            'id_numbers': r'\d{8,9}',  # ID numbers
        }
        
        key_concepts = []
        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, fodt_text)
            key_concepts.extend(matches)
        
        # Also extract important words (nouns, proper nouns)
        words = re.findall(r'\b[A-Za-z]{3,}\b', fodt_text)
        important_words = [word for word in words if len(word) > 4]
        key_concepts.extend(important_words[:10])  # Top 10 important words
        
        return key_concepts
    
    def find_best_matches(self, fodt_text, new_data: str, 
                         top_k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Find the best matching chunks from new_data for the given fodt_text
        
        Args:
            fodt_text: The FODT section to match against (string, list, or other)
            new_data: Your loaded data (string or list)
            top_k: Maximum number of results to return (optional)
            threshold: Minimum similarity score (0.0 to 1.0, optional)
            
        Returns:
            List of dictionaries with 'text', 'score', and 'concepts' keys
            
        Note: If both top_k and threshold are provided, both conditions must be met
        """
        
        # Handle different input types for fodt_text
        if isinstance(fodt_text, list):
            # If fodt_text is a list, join it into a single string
            fodt_text_str = "\n\n".join(str(item) for item in fodt_text)
        elif not isinstance(fodt_text, str):
            # Convert other types to string
            fodt_text_str = str(fodt_text)
        else:
            fodt_text_str = fodt_text
        
        # Default values
        if top_k is None and threshold is None:
            top_k = 3  # Default behavior
            
        # Chunk the new data
        chunks = self.chunk_new_data(new_data)
        
        if not chunks:
            return []
        
        # Extract key concepts from FODT text
        key_concepts = self.extract_key_concepts(fodt_text_str)
        
        # Create embeddings using OpenAI
        fodt_embedding = self.get_embeddings([fodt_text_str])
        chunk_embeddings = self.get_embeddings(chunks)
        
        # Calculate semantic similarity scores
        semantic_scores = cosine_similarity(fodt_embedding, chunk_embeddings)[0]
        
        # Calculate keyword-based scores
        keyword_scores = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            concept_matches = sum(1 for concept in key_concepts 
                                if concept.lower() in chunk_lower)
            keyword_score = concept_matches / max(len(key_concepts), 1)
            keyword_scores.append(keyword_score)
        
        # Combine scores (semantic similarity + keyword matching)
        combined_scores = []
        for i, (semantic, keyword) in enumerate(zip(semantic_scores, keyword_scores)):
            combined_score = 0.7 * semantic + 0.3 * keyword  # Weight semantic more
            combined_scores.append({
                'text': chunks[i],
                'score': combined_score,
                'semantic_score': semantic,
                'keyword_score': keyword,
                'matched_concepts': [c for c in key_concepts 
                                   if c.lower() in chunks[i].lower()]
            })
        
        # Sort by combined score
        best_matches = sorted(combined_scores, key=lambda x: x['score'], reverse=True)
        
        # Apply threshold filtering
        if threshold is not None:
            best_matches = [match for match in best_matches if match['score'] >= threshold]
            print(f"Found {len(best_matches)} matches above threshold {threshold}")
        
        # Apply top_k limit
        if top_k is not None:
            best_matches = best_matches[:top_k]
        
        return best_matches
    
    def find_specific_info(self, fodt_text: str, new_data: str, 
                          info_type: str) -> List[Dict[str, Any]]:
        """
        Find specific types of information (names, dates, amounts, etc.)
        
        Args:
            fodt_text: The FODT section to match against
            new_data: Your loaded data
            info_type: 'names', 'dates', 'amounts', 'addresses', etc.
        """
        
        # Enhance FODT text with info_type focus
        enhanced_query = f"{fodt_text} {info_type}"
        
        chunks = self.chunk_new_data(new_data)
        
        # Filter chunks that likely contain the requested info type
        type_patterns = {
            'names': r'[A-Z][a-z]+ [A-Z][a-z]+',
            'dates': r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}',
            'amounts': r'[\₪$€£]\s?\d+[,\d]*',
            'addresses': r'\d+\s+[A-Za-z\s]+',
            'phone': r'0\d{1,2}[-\s]?\d{7,8}',
            'case_info': r'case|court|legal|lawsuit'
        }
        
        pattern = type_patterns.get(info_type.lower(), r'.*')
        relevant_chunks = [chunk for chunk in chunks 
                          if re.search(pattern, chunk, re.IGNORECASE)]
        
        if not relevant_chunks:
            relevant_chunks = chunks  # Fallback to all chunks
        
        # Find best matches among relevant chunks
        best_matches = []
        if relevant_chunks:
            fodt_embedding = self.get_embeddings([enhanced_query])
            chunk_embeddings = self.get_embeddings(relevant_chunks)
            scores = cosine_similarity(fodt_embedding, chunk_embeddings)[0]
            
            for chunk, score in zip(relevant_chunks, scores):
                best_matches.append({
                    'text': chunk,
                    'score': score,
                    'info_type': info_type
                })
        
        return sorted(best_matches, key=lambda x: x['score'], reverse=True)[:3]

# Simple usage functions
def quick_match(fodt_text: str, new_data: str, openai_api_key: str, 
                model_name: str = 'text-embedding-3-small', top_k: int = 3) -> List[str]:
    """Quick function to get the best matching text chunks"""
    matcher = FODTTextMatcher(openai_api_key, model_name)
    matches = matcher.find_best_matches(fodt_text, new_data, top_k)
    return [match['text'] for match in matches]

def find_names(fodt_text: str, new_data: str, openai_api_key: str,
               model_name: str = 'text-embedding-3-small') -> List[str]:
    """Find name-related information from new_data"""
    matcher = FODTTextMatcher(openai_api_key, model_name)
    matches = matcher.find_specific_info(fodt_text, new_data, 'names')
    return [match['text'] for match in matches]

def find_amounts(fodt_text: str, new_data: str, openai_api_key: str,
                model_name: str = 'text-embedding-3-small') -> List[str]:
    """Find amount/financial information from new_data"""
    matcher = FODTTextMatcher(openai_api_key, model_name)
    matches = matcher.find_specific_info(fodt_text, new_data, 'amounts')
    return [match['text'] for match in matches]

# Usage Example
if __name__ == "__main__":
    # Example usage with your variables
    
    # Your OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Your FODT text section
    fodt_section = """
התובעות
    1. מוסיא אזרף, ת.ז 219180627,
הגדוד השלישי 54, צפת.
טל: 0546836018
    2.  חיה מושקא ברנשטיין, ת.ז 218568228
הקנאים 10/9, ערד.
טל:0546380901 
    3. סטרנה שרה עבדה, ת.ז 219177367
מושב גילת 416, ד.נ צפון הנגב.
טל: 0528027773 
    4. אלפרוביץ מרים, ת.ז 219123916
הקנאים 22, ערד.
טל: 0503901122 
    5. כמיה אגיב, ת.ז 333691566
מירון 35.
טל: 0523631236 
    6. מושקא שלג, ת.ז 334157229
לוי אשכול 82 דירה 2, תל אביב.
טל: 0525770750 
    7. אלרועי בתיה, ת.ז 217514603
לוי אשכול 24, תל אביב
טל: 0546836018
    8. אשכנזי חנה, ת.ז 217910405
זלמן שזר, צפת
טל: 0546596594
    9. פרגמנץ מושקא, ת.ז 334962701

    """
    
    NEW_DATA_DIR = Path("/home/tzuf/Desktop/projects/customized-LLM-experiments/examples/amit_test2/new_data/")


    
    # Initialize matcher
    matcher = FODTTextMatcher(OPENAI_API_KEY, 'text-embedding-3-small')

    # get new data
    new_data = matcher._load_new_data(NEW_DATA_DIR)

    # Find best matches
    matches = matcher.find_best_matches(fodt_section, new_data, top_k=2)
    
    # Or use simple functions:
    # best_texts = quick_match(fodt_section, new_data, OPENAI_API_KEY)
    # name_info = find_names(fodt_section, new_data, OPENAI_API_KEY)
    # amount_info = find_amounts(fodt_section, new_data, OPENAI_API_KEY)
    
    print("Matcher initialized successfully with OpenAI embeddings!")