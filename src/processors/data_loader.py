"""Data loading and cleaning utilities for contract generation."""

import logging
import re
from pathlib import Path
from typing import List

from ..utils.data_reader import DataReader
from ..utils.error_handler import ProcessingError

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and cleaning."""
    
    def __init__(self):
        self.data_reader = DataReader()
    
    def load_new_data(self, new_data_dir: str) -> str:
        """Load and combine new data from directory."""
        data_path = Path(new_data_dir)
        
        if not data_path.is_dir():
            raise ProcessingError(f"Data directory not found: {new_data_dir}")
        
        # Get supported file extensions
        supported_formats = self.data_reader.get_supported_formats()
        all_extensions = set()
        for format_list in supported_formats.values():
            all_extensions.update(format_list)
        
        # Find all supported files
        data_files = self._find_supported_files(data_path, all_extensions)
        
        if not data_files:
            raise ProcessingError(f"No supported data files found in: {new_data_dir}")
        
        # Process each file and combine data
        raw_contents = self._process_data_files(data_files)
        
        # Clean and combine content
        combined_content = "\n\n".join(raw_contents)
        cleaned_content = self.clean_data(combined_content)
        
        logger.info(f"Loaded and cleaned data: {len(cleaned_content)} characters")
        return cleaned_content
    
    def _find_supported_files(self, data_path: Path, supported_extensions: set) -> List[Path]:
        """Find all supported data files in directory."""
        data_files = []
        for file_path in data_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                data_files.append(file_path)
        return sorted(data_files)
    
    def _process_data_files(self, data_files: List[Path]) -> List[str]:
        """Process each data file and extract raw content."""
        raw_contents = []
        for file_path in data_files:
            try:
                file_data = self.data_reader.read_contract_data(str(file_path))
                if "raw_content" in file_data:
                    raw_contents.append(file_data['raw_content'])
            except Exception as e:
                logger.warning(f"Could not process file {file_path.name}: {e}")
                continue
        return raw_contents
    
    def clean_data(self, raw_data: str) -> str:
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
    
    def get_supported_formats(self) -> set:
        """Get all supported file formats."""
        supported_formats = self.data_reader.get_supported_formats()
        all_extensions = set()
        for format_list in supported_formats.values():
            all_extensions.update(format_list)
        return all_extensions
