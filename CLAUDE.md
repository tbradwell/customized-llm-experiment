# Claude Code Rules for Lawyer Contract Creation System

## Project Context
This is a quality-focused AI-powered legal contract generation system with comprehensive evaluation metrics, iterative refinement, and MLflow experiment tracking.

## Coding Standards

### Configuration Constants Rule
**CRITICAL: All configuration variables with specific numeric values MUST be defined at the top of files (below imports) in CAPITAL LETTERS.**

#### ✅ Correct Format:
```python
"""Module docstring."""

import logging
from typing import Dict, List
from config.settings import settings

# CONFIGURATION CONSTANTS  
MAX_COMPLETION_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5
DEFAULT_TEMPERATURE = 0.1
MIN_CONTENT_LENGTH = 100

logger = logging.getLogger(__name__)

class MyClass:
    def my_method(self):
        for i in range(MAX_COMPLETION_ITERATIONS):  # Use constant
            pass
```

#### ❌ Wrong Format:
```python
class MyClass:
    def my_method(self):
        for i in range(3):  # DON'T: Magic number
            pass
        
        max_iterations = 5  # DON'T: Should be at top in CAPITALS
```

### Why This Standard:
- **Visibility**: Easy to find and modify configuration values
- **Maintainability**: Single source of truth for constants  
- **Experimentation**: Simple to create variants with different parameters
- **Documentation**: Constants serve as self-documenting configuration

### Import Organization Rule
**CRITICAL: Imports MUST be at the very top of the page in alphabetical order, organized in two separate blocks:**

1. **External imports** (standard library + third-party packages) - alphabetical order
2. **Internal imports** (local project modules) - alphabetical order

#### ✅ Correct Import Format:
```python
"""Module docstring."""

# External imports (alphabetical order)
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import mlflow
import mlflow.tracking
from openai import OpenAI

# Internal imports (alphabetical order)
from config.settings import settings
from ..evaluation.metrics import MetricsCalculator
from ..utils.error_handler import ProcessingError, error_handler

# CONFIGURATION CONSTANTS
MAX_COMPLETION_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5

logger = logging.getLogger(__name__)
```

#### ❌ Wrong Import Format:
```python
"""Module docstring."""

# DON'T: Mixed external/internal, not alphabetical
from config.settings import settings
import json
from openai import OpenAI
import logging
from ..utils.error_handler import ProcessingError
```

### File Structure Order:
1. Module docstring
2. **External imports** (standard library + third-party) - alphabetical
3. **Internal imports** (local project modules) - alphabetical  
4. **CONFIGURATION CONSTANTS** (in CAPITAL_LETTERS)
5. Logger initialization  
6. Classes and functions

### Examples in Current Codebase:
- `src/core/content_generator.py`: `MAX_COMPLETION_ITERATIONS`, `MAX_REFINEMENT_ITERATIONS`
- `config/settings.py`: All threshold values and limits

## Project-Specific Guidelines

### Quality Metrics
- Always use the 7 PRD metrics: BLEU, ROUGE, METEOR, COMET, LLM Judge, Redundancy, Completeness
- Log all experiments to MLflow with code artifacts for reproducibility
- Include iterative refinement metadata in all evaluations

### Contract Generation
- Use iterative refinement until LLM says "NO_CHANGES_NEEDED"
- Ensure all placeholders (`{}`, `{{}}`, `[...]`, `...`) are completed
- Save complete code snapshots for experiment reproducibility

### MLflow Tracking
- Always log source code artifacts for reproducibility
- Include git information when available
- Use descriptive experiment names and tags
- Save generated contracts, skeletons, and quality reports

When making changes to this system, always follow these standards to maintain code quality and experimental reproducibility.