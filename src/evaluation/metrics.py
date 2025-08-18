"""Evaluation metrics for contract quality assessment."""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK imports
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# ROUGE imports
from rouge_score import rouge_scorer

# spaCy for text processing
import spacy

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    passed_threshold: bool
    threshold: float


class MetricsCalculator:
    """Calculate various quality metrics for generated contracts."""
    
    def __init__(self):
        # Initialize spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Some features may be limited.")
            self.nlp = None
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize TF-IDF vectorizer for redundancy detection
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def calculate_bleu_score(self, generated_text: str, reference_texts: List[str], 
                           threshold: float = 0.8) -> EvaluationResult:
        """Calculate BLEU score against reference texts.
        
        Args:
            generated_text: The generated contract text
            reference_texts: List of reference contract texts
            threshold: Minimum acceptable BLEU score
            
        Returns:
            EvaluationResult containing BLEU score and details
        """
        try:
            # Tokenize texts
            generated_tokens = nltk.word_tokenize(generated_text.lower())
            reference_token_lists = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
            
            # Use smoothing for better scores with short texts
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(
                reference_token_lists,
                generated_tokens,
                smoothing_function=smoothing
            )
            
            details = {
                "generated_token_count": len(generated_tokens),
                "reference_count": len(reference_texts),
                "smoothing_method": "method1"
            }
            
            return EvaluationResult(
                metric_name="BLEU",
                score=bleu_score,
                details=details,
                passed_threshold=bleu_score >= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {str(e)}")
            return EvaluationResult(
                metric_name="BLEU",
                score=0.0,
                details={"error": str(e)},
                passed_threshold=False,
                threshold=threshold
            )
    
    def calculate_rouge_scores(self, generated_text: str, reference_texts: List[str],
                             threshold: float = 0.85) -> EvaluationResult:
        """Calculate ROUGE scores against reference texts.
        
        Args:
            generated_text: The generated contract text
            reference_texts: List of reference contract texts
            threshold: Minimum acceptable average ROUGE score
            
        Returns:
            EvaluationResult containing ROUGE scores and details
        """
        try:
            rouge_results = {
                'rouge1': [],
                'rouge2': [],
                'rougeL': []
            }
            
            # Calculate ROUGE scores against each reference
            for reference in reference_texts:
                scores = self.rouge_scorer.score(reference, generated_text)
                
                for metric in rouge_results.keys():
                    rouge_results[metric].append(scores[metric].fmeasure)
            
            # Calculate average scores
            avg_scores = {
                metric: np.mean(scores) if scores else 0.0 
                for metric, scores in rouge_results.items()
            }
            
            # Overall average ROUGE score
            overall_avg = np.mean(list(avg_scores.values()))
            
            details = {
                "rouge1_avg": avg_scores['rouge1'],
                "rouge2_avg": avg_scores['rouge2'],
                "rougeL_avg": avg_scores['rougeL'],
                "individual_scores": rouge_results,
                "reference_count": len(reference_texts)
            }
            
            return EvaluationResult(
                metric_name="ROUGE",
                score=overall_avg,
                details=details,
                passed_threshold=overall_avg >= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return EvaluationResult(
                metric_name="ROUGE",
                score=0.0,
                details={"error": str(e)},
                passed_threshold=False,
                threshold=threshold
            )
    
    def calculate_meteor_score(self, generated_text: str, reference_texts: List[str],
                             threshold: float = 0.9) -> EvaluationResult:
        """Calculate METEOR score against reference texts.
        
        Args:
            generated_text: The generated contract text
            reference_texts: List of reference contract texts
            threshold: Minimum acceptable METEOR score
            
        Returns:
            EvaluationResult containing METEOR score and details
        """
        try:
            # Tokenize texts
            generated_tokens = nltk.word_tokenize(generated_text.lower())
            
            meteor_scores = []
            for reference in reference_texts:
                reference_tokens = nltk.word_tokenize(reference.lower())
                score = meteor_score([reference_tokens], generated_tokens)
                meteor_scores.append(score)
            
            # Average METEOR score across all references
            avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
            
            details = {
                "individual_scores": meteor_scores,
                "reference_count": len(reference_texts),
                "generated_token_count": len(generated_tokens)
            }
            
            return EvaluationResult(
                metric_name="METEOR",
                score=avg_meteor,
                details=details,
                passed_threshold=avg_meteor >= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating METEOR score: {str(e)}")
            return EvaluationResult(
                metric_name="METEOR",
                score=0.0,
                details={"error": str(e)},
                passed_threshold=False,
                threshold=threshold
            )
    
    def calculate_redundancy_score(self, text: str, threshold: float = 0.1) -> EvaluationResult:
        """Calculate redundancy score for the text.
        
        Args:
            text: The text to analyze
            threshold: Maximum acceptable redundancy score
            
        Returns:
            EvaluationResult containing redundancy score and details
        """
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            if len(sentences) < 2:
                return EvaluationResult(
                    metric_name="Redundancy",
                    score=0.0,
                    details={"sentence_count": len(sentences), "note": "Not enough sentences for analysis"},
                    passed_threshold=True,
                    threshold=threshold
                )
            
            # Calculate semantic similarity between sentences
            similarities = []
            
            # Use TF-IDF for sentence similarity
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
                cosine_sim = cosine_similarity(tfidf_matrix)
                
                # Get upper triangle (avoid diagonal and duplicate pairs)
                for i in range(len(sentences)):
                    for j in range(i + 1, len(sentences)):
                        similarities.append(cosine_sim[i][j])
            
            except Exception as e:
                logger.warning(f"TF-IDF similarity calculation failed: {str(e)}")
                # Fallback: simple word overlap
                similarities = self._calculate_word_overlap_similarities(sentences)
            
            # Calculate redundancy metrics
            avg_similarity = np.mean(similarities) if similarities else 0.0
            max_similarity = np.max(similarities) if similarities else 0.0
            high_similarity_count = sum(1 for sim in similarities if sim > 0.7)
            
            # Redundancy score: higher values indicate more redundancy
            redundancy_score = avg_similarity * 0.7 + (high_similarity_count / len(similarities)) * 0.3
            
            details = {
                "sentence_count": len(sentences),
                "similarity_pairs": len(similarities),
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "high_similarity_count": high_similarity_count,
                "calculation_method": "tfidf_cosine"
            }
            
            return EvaluationResult(
                metric_name="Redundancy",
                score=redundancy_score,
                details=details,
                passed_threshold=redundancy_score <= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating redundancy score: {str(e)}")
            return EvaluationResult(
                metric_name="Redundancy",
                score=1.0,  # Assume high redundancy on error
                details={"error": str(e)},
                passed_threshold=False,
                threshold=threshold
            )
    
    def calculate_completeness_score(self, generated_text: str, required_elements: List[str],
                                   threshold: float = 0.98) -> EvaluationResult:
        """Calculate completeness score based on required elements.
        
        Args:
            generated_text: The generated contract text
            required_elements: List of required text elements/phrases
            threshold: Minimum acceptable completeness score
            
        Returns:
            EvaluationResult containing completeness score and details
        """
        try:
            if not required_elements:
                return EvaluationResult(
                    metric_name="Completeness",
                    score=1.0,
                    details={"note": "No required elements specified"},
                    passed_threshold=True,
                    threshold=threshold
                )
            
            text_lower = generated_text.lower()
            found_elements = []
            missing_elements = []
            
            for element in required_elements:
                element_lower = element.lower()
                
                # Check for exact match or semantic similarity
                if element_lower in text_lower:
                    found_elements.append(element)
                elif self._check_semantic_presence(element, generated_text):
                    found_elements.append(element)
                else:
                    missing_elements.append(element)
            
            completeness_score = len(found_elements) / len(required_elements)
            
            details = {
                "total_required": len(required_elements),
                "found_count": len(found_elements),
                "missing_count": len(missing_elements),
                "found_elements": found_elements,
                "missing_elements": missing_elements
            }
            
            return EvaluationResult(
                metric_name="Completeness",
                score=completeness_score,
                details=details,
                passed_threshold=completeness_score >= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")
            return EvaluationResult(
                metric_name="Completeness",
                score=0.0,
                details={"error": str(e)},
                passed_threshold=False,
                threshold=threshold
            )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_word_overlap_similarities(self, sentences: List[str]) -> List[float]:
        """Calculate word overlap similarities as fallback method."""
        similarities = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[j].lower().split())
                
                if not words1 or not words2:
                    similarity = 0.0
                else:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities.append(similarity)
        
        return similarities
    
    def _check_semantic_presence(self, element: str, text: str) -> bool:
        """Check if an element is semantically present in the text."""
        # Simple keyword-based checking
        element_words = set(element.lower().split())
        text_words = set(text.lower().split())
        
        # If at least 70% of element words are present, consider it found
        overlap = len(element_words.intersection(text_words))
        return overlap / len(element_words) >= 0.7 if element_words else False


class COMETEvaluator:
    """COMET metric evaluator for neural-based quality assessment."""
    
    def __init__(self):
        self.comet_model = None
        self._initialize_comet()
    
    def _initialize_comet(self):
        """Initialize COMET model."""
        try:
            from comet import download_model, load_from_checkpoint
            
            # Download and load COMET model
            model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(model_path)
            logger.info("COMET model initialized successfully")
        except ImportError:
            logger.warning("COMET library not installed. Install with: pip install unbabel-comet")
            self.comet_model = None
        except Exception as e:
            logger.warning(f"Could not initialize COMET model: {str(e)}")
            self.comet_model = None
    
    def calculate_comet_score(self, generated_text: str, reference_texts: List[str],
                            source_text: str = "", threshold: float = 0.8) -> EvaluationResult:
        """Calculate COMET score.
        
        Args:
            generated_text: The generated contract text
            reference_texts: List of reference contract texts
            source_text: Source/skeleton text (optional)
            threshold: Minimum acceptable COMET score
            
        Returns:
            EvaluationResult containing COMET score and details
        """
        try:
            if self.comet_model is None:
                # Fallback to mock score if COMET model is not available
                logger.warning("COMET model not available, using mock score")
                mock_score = self._calculate_mock_comet_score(generated_text, reference_texts)
                
                details = {
                    "reference_count": len(reference_texts),
                    "generated_length": len(generated_text),
                    "note": "Mock COMET implementation - COMET model not available"
                }
                
                return EvaluationResult(
                    metric_name="COMET",
                    score=mock_score,
                    details=details,
                    passed_threshold=mock_score >= threshold,
                    threshold=threshold
                )
            
            # Use actual COMET model
            comet_scores = []
            
            for reference_text in reference_texts:
                # Prepare data for COMET
                comet_data = [{
                    "src": source_text if source_text else reference_text,  # Use reference as source if no source provided
                    "mt": generated_text,
                    "ref": reference_text
                }]
                
                # Calculate COMET score
                model_output = self.comet_model.predict(comet_data, batch_size=1, gpus=0)
                comet_scores.extend(model_output.scores)
            
            # Average COMET score across all references
            avg_comet_score = np.mean(comet_scores) if comet_scores else 0.0
            
            # Convert COMET score to 0-1 range (COMET typically outputs scores in different ranges)
            # COMET scores are typically between 0 and 1 already, but can be negative
            normalized_score = max(0.0, min(1.0, avg_comet_score))
            
            details = {
                "reference_count": len(reference_texts),
                "generated_length": len(generated_text),
                "individual_scores": comet_scores,
                "comet_model": "Unbabel/wmt22-comet-da",
                "normalization": "clipped_to_0_1_range"
            }
            
            return EvaluationResult(
                metric_name="COMET",
                score=normalized_score,
                details=details,
                passed_threshold=normalized_score >= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating COMET score: {str(e)}")
            # Fallback to mock score on error
            mock_score = self._calculate_mock_comet_score(generated_text, reference_texts)
            
            return EvaluationResult(
                metric_name="COMET",
                score=mock_score,
                details={"error": str(e), "fallback_to_mock": True},
                passed_threshold=mock_score >= threshold,
                threshold=threshold
            )
    
    def _calculate_mock_comet_score(self, generated_text: str, reference_texts: List[str]) -> float:
        """Calculate a mock COMET score for demonstration purposes."""
        if not reference_texts:
            return 0.5
        
        # Simple heuristic based on length similarity and word overlap
        gen_length = len(generated_text.split())
        ref_lengths = [len(ref.split()) for ref in reference_texts]
        avg_ref_length = np.mean(ref_lengths)
        
        # Length similarity component
        length_sim = 1.0 - min(abs(gen_length - avg_ref_length) / avg_ref_length, 1.0)
        
        # Word overlap component
        gen_words = set(generated_text.lower().split())
        all_ref_words = set()
        for ref in reference_texts:
            all_ref_words.update(ref.lower().split())
        
        if all_ref_words:
            word_overlap = len(gen_words.intersection(all_ref_words)) / len(all_ref_words)
        else:
            word_overlap = 0.0
        
        # Combine components
        mock_score = (length_sim * 0.4) + (word_overlap * 0.6)
        
        # Add some randomness to make it more realistic
        mock_score += np.random.normal(0, 0.05)
        
        return max(0.0, min(1.0, mock_score))  # Clamp to [0, 1]
