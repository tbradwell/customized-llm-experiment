"""Simple pipeline for skeleton-based contract generation with MLflow tracking."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import mlflow
import mlflow.tracking
from docx import Document
from openai import OpenAI

from config.settings import settings
from ..evaluation.metrics import MetricsCalculator
from ..utils.doc_handler import DocHandler
from ..utils.error_handler import ProcessingError, error_handler

# CONFIGURATION CONSTANTS
MAX_ITERATIONS = 10
DEFAULT_TEMPERATURE = 0.1
MIN_CONTENT_LENGTH = 50

logger = logging.getLogger(__name__)


class SimplePipeline:
    """Simple pipeline for contract generation using skeleton replacement."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = DEFAULT_TEMPERATURE
        self.metrics_calc = MetricsCalculator()
        self.paragraph_block = []
    
    def _setup_mlflow(self):
        """Setup MLflow tracking with experiment creation."""
        try:
            # Set MLflow tracking URI to local directory
            tracking_uri = "./mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            
            # Get or create experiment named "contract_generation"
            experiment_name = "contract_generation"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    # Create new experiment
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created MLflow experiment '{experiment_name}' with ID: {experiment_id}")
                else:
                    logger.info(f"Using existing MLflow experiment '{experiment_name}' with ID: {experiment.experiment_id}")
            except Exception as create_error:
                logger.warning(f"Could not create/get experiment: {create_error}")
                # Use default experiment
                experiment_name = "Default"
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            # Fallback: just set tracking URI and let MLflow handle the rest
            mlflow.set_tracking_uri("./mlruns")
        
    def run_pipeline(self, new_data_dir: str, skeleton_path: str, output_dir: str,
                    contract_type: str = "legal_claim", test_type: str = "amit_example") -> Dict[str, Any]:
        """Run the complete pipeline with MLflow tracking.
        
        Args:
            new_data_dir: Directory containing new data files
            skeleton_path: Path to skeleton document
            output_dir: Directory to save outputs
            contract_type: Type of contract being generated
            test_type: Type of test for MLflow tagging
            
        Returns:
            Dictionary with pipeline results
        """
        # Setup MLflow
        self._setup_mlflow()
        
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            try:
                # Set MLflow tags
                mlflow.set_tag("approach", "simple_skeleton")
                mlflow.set_tag("test_type", test_type)
                mlflow.set_tag("status", "RUNNING")
                
                # Set MLflow parameters
                mlflow.log_param("contract_type", contract_type)
                mlflow.log_param("openai_model", self.model)
                
                # Step 1: Load new data
                logger.info("Step 1: Loading new data")
                new_data = self._load_new_data(new_data_dir)
                mlflow.log_param("new_data_length", len(new_data))
                
                # Step 2: Load skeleton
                logger.info("Step 2: Loading skeleton document")
                skeleton_doc = Document(skeleton_path)
                
                # Count initial placeholders
                initial_text = DocHandler.extract_text_from_doc(skeleton_doc)
                initial_placeholders = len(re.findall(r'\{[^}]*\}', initial_text))
                mlflow.log_metric("initial_placeholders", initial_placeholders)
                
                # Step 3: Process skeleton iteratively
                logger.info("Step 3: Processing skeleton paragraphs")
                processed_doc, iterations = self._process_skeleton_iteratively(
                    skeleton_doc, new_data
                )
                
                mlflow.log_metric("completion_iterations", iterations)
                
                # Step 4: Verify completion
                final_text = DocHandler.extract_text_from_doc(processed_doc)
                final_placeholders = len(re.findall(r'\{[^}]*\}', final_text))
                mlflow.log_metric("final_placeholders", final_placeholders)
                
                # Set completion status
                completion_status = "COMPLETE" if final_placeholders == 0 else "INCOMPLETE"
                mlflow.log_param("completion_status", completion_status)
                
                # Step 5: Save outputs and log artifacts
                logger.info("Step 5: Saving outputs and logging artifacts")
                artifacts_info = self._save_outputs_and_artifacts(
                    processed_doc, skeleton_path, final_text, run_id, output_dir
                )
                
                # Step 6: Run evaluation if ground truth exists
                logger.info("Step 6: Running evaluation")
                evaluation_results = self._run_evaluation(
                    final_text, output_dir, artifacts_info
                )
                
                # Log evaluation metrics
                if evaluation_results:
                    for metric_name, score in evaluation_results.items():
                        if isinstance(score, (int, float)):
                            mlflow.log_metric(f"quality_{metric_name}", score)
                
                # Set final status
                mlflow.set_tag("status", "FINISHED")
                
                # Create results summary
                results = {
                    "run_id": run_id,
                    "success": True,
                    "initial_placeholders": initial_placeholders,
                    "final_placeholders": final_placeholders,
                    "completion_iterations": iterations,
                    "completion_status": completion_status,
                    "final_content_length": len(final_text),
                    "evaluation_results": evaluation_results,
                    "artifacts": artifacts_info
                }
                
                logger.info(f"Pipeline completed successfully. Run ID: {run_id}")
                return results
                
            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error", str(e))
                raise ProcessingError(f"Pipeline failed: {str(e)}")
    
    def _load_new_data(self, new_data_dir: str) -> str:
        """Load and combine new data from directory."""
        from ..utils.data_reader import DataReader
        
        data_reader = DataReader()
        data_path = Path(new_data_dir)
        
        if not data_path.is_dir():
            raise ProcessingError(f"Data directory not found: {new_data_dir}")
        
        # Get supported file extensions
        supported_formats = data_reader.get_supported_formats()
        all_extensions = set()
        for format_list in supported_formats.values():
            all_extensions.update(format_list)
        
        # Find all supported files
        data_files = []
        for file_path in data_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                data_files.append(file_path)
        
        if not data_files:
            raise ProcessingError(f"No supported data files found in: {new_data_dir}")
        
        # Process each file and combine data
        raw_contents = []
        for file_path in sorted(data_files):
            try:
                file_data = data_reader.read_contract_data(str(file_path))
                if "raw_content" in file_data:
                    raw_contents.append(file_data['raw_content'])
            except Exception as e:
                logger.warning(f"Could not process file {file_path.name}: {e}")
                continue
        
        # Clean and combine content
        combined_content = "\n\n".join(raw_contents)
        cleaned_content = self._clean_data(combined_content)
        
        logger.info(f"Loaded and cleaned data: {len(cleaned_content)} characters")
        return cleaned_content
    
    def _clean_data(self, raw_data: str) -> str:
        """Clean the data by removing document headers and unwanted formatting."""
        if not raw_data:
            return ""
        
        # Remove document name headers
        cleaned = re.sub(r'=== [^=]+\.(docx|pdf|msg|jpg|jpeg|png|tiff|bmp) ===\n?', '', raw_data)
        
        # Remove document references
        cleaned = re.sub(r'DOC-\d{8}-[A-Z0-9]+\.(docx|pdf|msg|jpg|jpeg|png|tiff|bmp)', '', cleaned)
        cleaned = re.sub(r'DOC-[^\s\n]+', '', cleaned)
        
        # Clean up whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    

    def _process_skeleton_iteratively(self, skeleton_doc: Document, new_data: str) -> Tuple[Document, int]:
        """Process skeleton document iteratively using phase-based approach."""
        
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Processing iteration {iteration + 1}/{MAX_ITERATIONS}")
            
            changes_made = False
            
            # Execute the 4 phases in order
            regular_paragraphs, template_blocks = self._phase1_identify_work(skeleton_doc)
            
            template_changes = self._phase2_process_templates(skeleton_doc, template_blocks, new_data)
            if template_changes:
                changes_made = True
            
            regular_changes = self._phase3_process_regular_paragraphs(skeleton_doc, regular_paragraphs, new_data)
            if regular_changes:
                changes_made = True
            
            table_changes = self._phase4_process_tables(skeleton_doc, new_data)
            if table_changes:
                changes_made = True
            
            # Check completion
            current_text = DocHandler.extract_text_from_doc(skeleton_doc)
            remaining_placeholders = re.findall(r'\{[^}]*\}', current_text)
            
            logger.info(f"Iteration {iteration + 1} complete. Remaining placeholders: {len(remaining_placeholders)}")
            
            if not remaining_placeholders:
                logger.info(f"All placeholders resolved after {iteration + 1} iterations")
                break
            
            if not changes_made:
                logger.warning(f"No changes made in iteration {iteration + 1}, stopping")
                break
        
        return skeleton_doc, iteration + 1


    def _phase1_identify_work(self, skeleton_doc: Document) -> Tuple[List[Dict], List[Dict]]:
        """
        Phase 1: Identify all work that needs to be done without making any modifications.
        
        Returns:
            Tuple of (regular_paragraphs, template_blocks)
        """
        logger.info("Phase 1: Identifying work to be done...")
        
        regular_paragraphs = []      # Paragraphs with simple placeholders like {name}
        template_blocks = []         # Template blocks with {% %} that need duplication
        
        # Reset paragraph block tracking
        self.paragraph_block = []
        current_block = []
        block_start_idx = None
        
        # Scan through document to identify what needs processing
        for i, paragraph in enumerate(skeleton_doc.paragraphs):
            if not paragraph.text.strip():
                continue
            
            # Check for {% %} block placeholders - template loops that need multiple paragraphs
            if '%}' in paragraph.text and '{%' not in paragraph.text:
                # Starting a template block
                if block_start_idx is None:
                    block_start_idx = i
                current_block.append(paragraph.text)
                
            elif '{%' in paragraph.text:
                # Ending a template block
                current_block.append(paragraph.text)
                if current_block and block_start_idx is not None:
                    template_blocks.append({
                        'start_index': block_start_idx,
                        'paragraph_texts': current_block.copy(),
                        'block_size': len(current_block)
                    })
                # Reset for next block
                current_block = []
                block_start_idx = None
                
            elif len(current_block) > 0:
                # Continue building current template block
                current_block.append(paragraph.text)
                
            elif '}' in paragraph.text:
                # Regular placeholder paragraph - store additional identifying info
                regular_paragraphs.append({
                    'index': i,
                    'original_text': paragraph.text,  # Store original text for comparison
                    'text_hash': hash(paragraph.text.strip()),  # Hash for quick comparison
                    'char_count': len(paragraph.text.strip()),  # Length for validation
                    'has_placeholders': True  # Flag to indicate this needs processing
                })
        
        logger.info(f"Found {len(regular_paragraphs)} regular paragraphs and {len(template_blocks)} template blocks")
        
        return regular_paragraphs, template_blocks


    def _phase2_process_templates(self, skeleton_doc: Document, template_blocks: List[Dict], new_data: str) -> bool:
        """
        Phase 2: Process template blocks that need duplication.
        
        Args:
            skeleton_doc: Document to process
            template_blocks: List of template block info from phase 1
            new_data: Data for processing
            
        Returns:
            bool: True if any changes were made
        """
        logger.info("Phase 2: Processing template blocks...")
        
        changes_made = False
        
        # Process template blocks in reverse order so that earlier insertions 
        # don't affect the indices of later blocks
        for block_info in reversed(template_blocks):
            
            # Set up paragraph_block for your existing _process_template_block method
            self.paragraph_block = block_info['paragraph_texts'].copy()
            
            # Use your existing template processing method
            list_changes, block_changed = self._process_template_block(new_data)
            
            if block_changed:
                start_idx = block_info['start_index']
                block_size = block_info['block_size']
                
                logger.info(f"Processing template block at index {start_idx}, size {block_size}")
                
                # Clear existing paragraphs in the template block
                DocHandler.clear_template_block_paragraphs(skeleton_doc, start_idx, block_size)
                
                # Store references and insert additional blocks
                reference_paragraphs = DocHandler.get_reference_paragraphs(skeleton_doc, start_idx, block_size)
                DocHandler.insert_additional_template_blocks(skeleton_doc, start_idx, reference_paragraphs, list_changes)
                
                # Add text content to all blocks
                DocHandler.add_text_to_template_blocks(skeleton_doc, start_idx, list_changes)
                
                changes_made = True
        
        # Reset paragraph_block after template processing
        self.paragraph_block = []
        
        return changes_made


    def _phase3_process_regular_paragraphs(self, skeleton_doc: Document, regular_paragraphs: List[Dict], new_data: str) -> bool:
        """
        Phase 3: Process regular paragraphs with simple placeholders.
        
        Args:
            skeleton_doc: Document to process
            regular_paragraphs: List of regular paragraph info from phase 1
            new_data: Data for processing
            
        Returns:
            bool: True if any changes were made
        """
        logger.info("Phase 3: Processing regular paragraphs...")
        
        changes_made = False
        
        for para_info in regular_paragraphs:
            original_index = para_info['index']
            original_text = para_info['original_text']
            text_hash = para_info['text_hash']
            char_count = para_info['char_count']
            
            # First try to find paragraph at original position with safe text comparison
            target_paragraph = None
            
            if (original_index < len(skeleton_doc.paragraphs)):
                current_para = skeleton_doc.paragraphs[original_index]
                # Safe comparison using text content and validation
                if (current_para.text.strip() == original_text.strip() and 
                    len(current_para.text.strip()) == char_count and
                    hash(current_para.text.strip()) == text_hash):
                    target_paragraph = current_para
                    logger.debug(f"Found paragraph at original index {original_index}")
            
            # If not found at original position, search through document
            if target_paragraph is None:
                logger.info(f"Paragraph moved from index {original_index}, searching by content...")
                for i, paragraph in enumerate(skeleton_doc.paragraphs):
                    if (paragraph.text.strip() == original_text.strip() and 
                        len(paragraph.text.strip()) == char_count and
                        hash(paragraph.text.strip()) == text_hash):
                        target_paragraph = paragraph
                        logger.debug(f"Found moved paragraph at index {i}")
                        break
            
            # Process the paragraph if found
            if target_paragraph:
                if self._process_single_paragraph(target_paragraph, new_data):
                    changes_made = True
            else:
                logger.warning(f"Could not locate paragraph with text: {original_text[:50]}...")
        
        return changes_made


    def _phase4_process_tables(self, skeleton_doc: Document, new_data: str) -> bool:
        """
        Phase 4: Process table cell paragraphs.
        
        Args:
            skeleton_doc: Document to process
            new_data: Data for processing
            
        Returns:
            bool: True if any changes were made
        """
        logger.info("Phase 4: Processing tables...")
        
        changes_made = False
        
        for table in skeleton_doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        if '{' in paragraph.text:
                            new_text, paragraph_changed = self._process_paragraph(
                                paragraph.text, new_data
                            )
                            
                            if paragraph_changed:
                                if DocHandler.replace_paragraph_text(paragraph, new_text):
                                    logger.info(f"Updated paragraph to: {new_text[:100]}...")
                                    return True
                                else:
                                    logger.warning("Failed to replace paragraph text")
        
        
            
        return changes_made


    def _find_and_process_moved_paragraph(self, skeleton_doc: Document, target_text: str, text_hash: int, char_count: int, new_data: str) -> bool:
        """
        Helper method to find and process a paragraph that may have moved due to template processing.
        
        Args:
            skeleton_doc: Document to search
            target_text: Original text to find
            text_hash: Hash of the text for validation
            char_count: Character count for validation
            new_data: Data for processing
            
        Returns:
            bool: True if paragraph was found and processed
        """
        for paragraph in skeleton_doc.paragraphs:
            if (paragraph.text.strip() == target_text.strip() and 
                len(paragraph.text.strip()) == char_count and
                hash(paragraph.text.strip()) == text_hash):
                return self._process_single_paragraph(paragraph, new_data)
        
        return False

    def _process_paragraph(self, paragraph_text: str, new_data: str) -> Tuple[str, bool]:
        """Process a single paragraph containing placeholders.
        
        Args:
            paragraph_text: Text of the paragraph to process
            new_data: New data to use for filling placeholders
            
        Returns:
            Tuple of (new_text, changed)
        """
        
        
        # Check for {~~} placeholders - these can be deleted if no suitable information
        if '{~' in paragraph_text and '~}' in paragraph_text:
            return self._process_deletable_paragraph(paragraph_text, new_data)
        
        # Check for regular placeholders
        elif '{' in paragraph_text and '}' in paragraph_text:
            return self._process_fillable_paragraph(paragraph_text, new_data)
        
        return paragraph_text, False
    
    def _process_template_block(self, new_data:str) -> Tuple[list[str], bool]:
        """Process template block with {% %} placeholders that contain multiple paragraphs.
        
        This method handles template blocks like:
        {% for p in plaintiffs %}
        {p.PERSON_FULL_NAME}, ת.ז. {p.ID}
        {p.ADDRESS}
        טל: {p.PHONE}; פקס: {p.FAX}
        {% endfor %}
        """
        
       
        # Process the paragraphs iteratively with the LLM
        filled_content = self._fill_template_block(self.paragraph_block, new_data)
        

        return filled_content, True if len(filled_content)>0 else False
        
    def _find_and_process_moved_paragraph(self, skeleton_doc: Document, paragraph, new_data: str) -> bool:
        """
        Find and process a paragraph that moved due to template processing.
        
        Returns:
            bool: True if changes were made
        """
        for doc_paragraph in skeleton_doc.paragraphs:
            if doc_paragraph == paragraph:
                new_text, paragraph_changed = self._process_paragraph(paragraph.text, new_data)
                
                if paragraph_changed:
                    if DocHandler.replace_paragraph_text(paragraph, new_text):
                        logger.info(f"Updated paragraph to: {new_text[:100]}...")
                        return True
                    else:
                        logger.warning("Failed to replace paragraph text")
        
                break
        
        return False

    def _process_single_paragraph(self, paragraph, new_data: str) -> bool:
        """
        Process a single paragraph with placeholders.
        
        Returns:
            bool: True if changes were made
        """
        logger.info(f"Processing paragraph: {paragraph.text[:100]}...")
        
        new_text, paragraph_changed = self._process_paragraph(paragraph.text, new_data)
        
        if paragraph_changed:
            if DocHandler.replace_paragraph_text(paragraph, new_text):
                logger.info(f"Updated paragraph to: {new_text[:100]}...")
                return True
            else:
                logger.warning("Failed to replace paragraph text")
        
        
        return False


    def _extract_template_blocks(self, paragraph_list: List[str]) -> List[Tuple[str, str]]:
        """Extract and clean paragraphs from template block list.
        
        Args:
            paragraph_list: List of paragraph texts containing template syntax
            
        Returns:
            List of tuples (cleaned_content, original_content) 
        """
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
    
    
    def _fill_template_block(self, paragraphs: List[str], new_data: str) -> str:
        """Fill template paragraphs iteratively until LLM says no more to fill."""
        
        # Join paragraphs with separation markers
        separation_marker_par = "---PARAGRAPH_SEPARATOR---"
        separation_marker_block = "---BLOCK---"
    
        combined_paragraphs = f"\n{separation_marker_par}\n".join(paragraphs)
        
        filled_paragraphs = combined_paragraphs
        
        
        # Send to LLM for filling
        new_content = self._send_paragraphs_to_llm(
            filled_paragraphs, new_data, separation_marker_par, separation_marker_block)
        
           
      
        # Remove separation markers from final result
        result = new_content.replace(f"\n{separation_marker_par}\n", "\n").replace(f"{separation_marker_par}", "\n")
        # result = re.sub(r'\n+', '\n', result)
        result_blocks = result.split(separation_marker_block)

        clean_block = []
        for block in result_blocks:
            clean_pargraphs = []
            pargraphs = block.split('\n')
            for pargraph in pargraphs:
                new_paragraph = pargraph.replace('%}', '').replace('{%', '')
                if new_paragraph!='':
                    clean_pargraphs.append(new_paragraph)
            if len(clean_pargraphs)>0:
                clean_block.append(clean_pargraphs)

        return clean_block
    
    def _send_paragraphs_to_llm(
            self, 
            paragraphs_text: str, 
            new_data: str, 
            par_separator: str,
            block_separator) -> str:
        """Send paragraphs to LLM for filling with proper separation handling."""
        
        prompt = f"""
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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert legal document completion specialist. Be precise and factual."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to process template paragraphs: {e}")
            return ""
    
    def _process_deletable_paragraph(self, paragraph_text: str, new_data: str) -> Tuple[str, bool]:
        """Process paragraph with {~~} placeholders that can be deleted."""
        
        prompt = f"""
You are a legal document editor. The paragraph below contains {{~~}} placeholders that can be deleted if no suitable information exists in the new data.

PARAGRAPH TO PROCESS:
{paragraph_text}

NEW DATA AVAILABLE:
{new_data[:1500]}

INSTRUCTIONS:
1. If the NEW DATA contains relevant information for the {{~~}} placeholder, fill it with specific details
2. If the NEW DATA does not contain relevant information, DELETE the entire paragraph by returning empty string
3. Do not use generic text like "יש לציין" or "יינתן במועד מאוחר יותר"
4. Use only specific, factual information from the NEW DATA

Return the processed paragraph (or empty string to delete):
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert legal document editor. Be precise and factual."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

            )
            
            new_text = response.choices[0].message.content.strip()
            
            # Check if significantly different
            if new_text != paragraph_text:
                return new_text, True
            
            return paragraph_text, False
            
        except Exception as e:
            logger.error(f"Failed to process deletable paragraph: {e}")
            return paragraph_text, False
    
    def _process_fillable_paragraph(self, paragraph_text: str, new_data: str) -> Tuple[str, bool]:
        """Process paragraph with regular {} placeholders."""
        
        prompt = f"""
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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert legal document completion specialist."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            
            new_text = response.choices[0].message.content.strip()
            
            # Check if placeholders were actually filled
            remaining_placeholders = re.findall(r'\{[^}]*\}', new_text)
            original_placeholders = re.findall(r'\{[^}]*\}', paragraph_text)
            
            if len(remaining_placeholders) < len(original_placeholders):
                return new_text, True
            
            return paragraph_text, False
            
        except Exception as e:
            logger.error(f"Failed to process fillable paragraph: {e}")
            return paragraph_text, False
    
    def _save_outputs_and_artifacts(self, processed_doc: Document, skeleton_path: str,
                                   final_text: str, run_id: str, output_dir: str) -> Dict[str, str]:
        """Save outputs and log MLflow artifacts."""
        
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save final contract
        contract_path = output_path / "contract_simple.docx"
        processed_doc.save(contract_path)
        
        # Copy original skeleton for reference
        skeleton_copy_path = output_path / "sekeleton_oracle.docx"
        import shutil
        shutil.copy2(skeleton_path, skeleton_copy_path)
        
        # Create quality assessment report
        quality_report = {
            "final_content_length": len(final_text),
            "placeholder_analysis": {
                "remaining_placeholders": len(re.findall(r'\{[^}]*\}', final_text)),
                "placeholder_examples": re.findall(r'\{[^}]*\}', final_text)[:5]
            },
            "content_analysis": {
                "has_real_names": any(name in final_text for name in ["זואי", "מור", "גיל"]),
                "has_real_address": "חיים הרצוג" in final_text,
                "has_real_phone": "0542477683" in final_text,
                "unwanted_content": {
                    "doc_references": "DOC-" in final_text,
                    "generic_placeholders": "יינתן במועד מאוחר יותר" in final_text
                }
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        quality_report_path = output_path / "quality_assessment.json"
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(str(contract_path), "contracts")
        mlflow.log_artifact(str(skeleton_copy_path), "skeletons")
        mlflow.log_artifact(str(quality_report_path), "reports")
        
        return {
            "contract_path": str(contract_path),
            "skeleton_path": str(skeleton_copy_path),
            "quality_report_path": str(quality_report_path)
        }
    
    def _run_evaluation(self, final_text: str, output_dir: str, artifacts_info: Dict[str, str]) -> Dict[str, float]:
        """Run evaluation metrics against ground truth if available."""
        
        # Look for ground truth file
        gt_path = Path(output_dir).parent / "gt.docx"
        if not gt_path.exists():
            logger.info("No ground truth file found, skipping evaluation")
            return {}
        
        try:
            # Extract ground truth text
            gt_text = DocHandler.extract_text_from_docx(str(gt_path))
            
            # Calculate metrics
            metrics = {}
            reference_texts = [gt_text]
            
            # BLEU Score
            try:
                bleu_result = self.metrics_calc.calculate_bleu_score(final_text, reference_texts)
                metrics['bleu'] = bleu_result.score
            except Exception as e:
                logger.warning(f"BLEU calculation failed: {e}")
                metrics['bleu'] = 0.0
            
            # ROUGE Scores
            try:
                rouge_result = self.metrics_calc.calculate_rouge_scores(final_text, reference_texts)
                metrics['rouge'] = rouge_result.details.get('rougeL_f', 0.0)
            except Exception as e:
                logger.warning(f"ROUGE calculation failed: {e}")
                metrics['rouge'] = 0.0
            
            # METEOR Score
            try:
                meteor_result = self.metrics_calc.calculate_meteor_score(final_text, reference_texts)
                metrics['meteor'] = meteor_result.score
            except Exception as e:
                logger.warning(f"METEOR calculation failed: {e}")
                metrics['meteor'] = 0.0
            
            # LLM Judge Score
            try:
                llm_result = self.metrics_calc.calculate_llm_judge_score(final_text, reference_texts)
                metrics['llm_judge'] = llm_result.score
            except Exception as e:
                logger.warning(f"LLM Judge calculation failed: {e}")
                metrics['llm_judge'] = 0.0
            
            logger.info(f"Evaluation completed with {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}