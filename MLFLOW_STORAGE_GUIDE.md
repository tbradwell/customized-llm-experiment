# MLflow Storage Guide for Contract Generation System

## 📍 Storage Locations

### Primary Storage
```
/home/tzuf/Desktop/projects/cutomizd-LLM-experiments/
├── mlflow.db                    # SQLite database (metadata)
└── mlruns/                      # Artifacts directory
    └── 0/                       # Experiment ID (contract_generation)
        └── {run_id}/            # Individual run folder
            ├── meta.yaml        # Run metadata
            ├── metrics/         # All numeric metrics
            ├── params/          # All parameters
            ├── tags/           # All tags
            └── artifacts/      # Files and documents
                ├── contracts/
                ├── skeletons/
                ├── reports/
                ├── evaluation_details/
                ├── source_code/        # 🆕 Core source files
                ├── code_snapshot/      # 🆕 Complete codebase copy
                └── environment/        # 🆕 Dependencies & versions
```

## 💾 What Gets Saved for Each Experiment

### 📊 Metrics (Numeric Values)
- **Quality Scores**: `quality_bleu`, `quality_rouge`, `quality_comet`, `quality_llm_judge`
- **Iteration Details**: `completion_iterations`, `initial_placeholders`, `final_placeholders`, `refinement_iterations`
- **Performance**: `generation_time_seconds`, `prompt_tokens`, `completion_tokens`
- **Quality Gates**: `gate_bleu_score`, `gate_rouge_score`, etc.

### 🏷️ Parameters (Configuration Values)
- **Contract Info**: `contract_type`, `client_name`, `provider_name`
- **Model Config**: `openai_model`, `temperature`, `quality_gates_enabled`
- **Iteration Results**: `final_decision`, `refinement_complete`

### 🔖 Tags (Custom Labels)
- **Approach**: `approach=with_iteration` or `approach=without_iteration`
- **Test Type**: `test_type=amit_example`
- **Version**: `version=1.0`
- **Features**: `feature=placeholder_completion`

### 📎 Artifacts (Files)
1. **Generated Contracts**
   - Path: `artifacts/contracts/contract_with_iteration.docx`
   - Contains: Final generated legal document

2. **Skeleton Templates**
   - Path: `artifacts/skeletons/sekeleton_oracle.docx`
   - Contains: Original template with placeholders

3. **Quality Reports**
   - Path: `artifacts/reports/temp_quality_report.json`
   - Contains: Comprehensive quality assessment

4. **Evaluation Details**
   - Path: `artifacts/evaluation_details/`
   - Files:
     - `temp_bleu_details.json` - BLEU score breakdown
     - `temp_rouge_details.json` - ROUGE score details
     - `temp_comet_details.json` - COMET evaluation results
     - `temp_llm_judge_details.json` - LLM Judge detailed feedback

5. **🆕 SOURCE CODE** (Complete Reproducibility)
   - Path: `artifacts/source_code/`
   - Files:
     - `src/core/content_generator.py` - Iterative refinement logic
     - `src/core/quality_pipeline.py` - Pipeline orchestration
     - `src/core/document_processor.py` - Document handling
     - `src/evaluation/metrics.py` - Quality evaluation
     - `src/evaluation/llm_judge.py` - LLM assessment
     - `config/settings.py` - System configuration

6. **🆕 COMPLETE CODE SNAPSHOT**
   - Path: `artifacts/code_snapshot/`
   - Contains: Entire codebase copy with file hashes
   - Files:
     - `src/` - Complete source directory
     - `config/` - Configuration files
     - `examples/` - Example scripts
     - `requirements.txt` - Dependencies
     - `code_manifest.json` - File hashes & metadata

7. **🆕 ENVIRONMENT INFO**
   - Path: `artifacts/environment/`
   - Files:
     - `requirements.txt` - Exact dependency versions
   - Parameters:
     - `python_version`, `platform`, `architecture`
     - `package_openai_version`, `package_nltk_version`, etc.

8. **🆕 GIT VERSION CONTROL**
   - Parameters:
     - `git_commit_hash` - Exact code version
     - `git_branch` - Development branch
     - `git_clean` - Uncommitted changes check
     - `git_commit_message` - What was changed
     - `git_author` - Who made changes

## 🧪 Example: Your Iterative Refinement Experiment

When you run:
```python
pipeline.process_contract(
    skeleton_path='examples/amit_test/sekeleton_oracle.docx',
    contract_data=contract_data,
    experiment_name='iterative_refinement_test',
    experiment_tags={'approach': 'with_iteration', 'test_type': 'amit_example'}
)
```

MLflow saves:
```
mlruns/0/{unique_run_id}/
├── meta.yaml
├── metrics/
│   ├── quality_bleu (0.0374)
│   ├── quality_rouge (0.3443)
│   ├── quality_comet (0.8473)
│   ├── quality_llm_judge (3.65)
│   ├── completion_iterations (2)
│   ├── initial_placeholders (48)
│   ├── final_placeholders (0)
│   └── refinement_iterations (1)
├── params/
│   ├── contract_type (legal_claim)
│   ├── final_decision (NO_CHANGES_NEEDED)
│   ├── refinement_complete (True)
│   └── openai_model (gpt-5)
├── tags/
│   ├── approach (with_iteration)
│   ├── test_type (amit_example)
│   └── status (FINISHED)
└── artifacts/
    ├── contracts/
    │   └── contract_with_iteration.docx (3,254 chars)
    ├── skeletons/
    │   └── sekeleton_oracle.docx (original template)
    └── reports/
        └── quality_assessment.json (detailed analysis)
```

## 🔍 How to Access Your Experiments

### 1. MLflow UI (Recommended)
```bash
cd /home/tzuf/Desktop/projects/cutomizd-LLM-experiments
mlflow ui
# Open: http://localhost:5000
```

### 2. Direct File Access
```bash
# List all experiments
ls mlruns/0/

# Access specific run artifacts
ls mlruns/0/{run_id}/artifacts/contracts/

# View generated contract
open mlruns/0/{run_id}/artifacts/contracts/contract_with_iteration.docx
```

### 3. Programmatic Access
```python
import mlflow
runs = mlflow.search_runs(experiment_ids=["0"])
run_id = runs.iloc[0]['run_id']
contract_path = f"mlruns/0/{run_id}/artifacts/contracts/contract_with_iteration.docx"
```

## 📈 Comparing Experiments

Your experiments will be automatically organized for comparison:

| Experiment Name | Approach | BLEU | ROUGE | COMET | LLM Judge | Iterations |
|----------------|----------|------|-------|-------|-----------|------------|
| iterative_test | with_iteration | 0.0374 | 0.3443 | 0.8473 | 3.65 | 1 |
| baseline_test | without_iteration | 0.1812 | 0.2094 | 0.7321 | 2.35 | 0 |

## 🎯 Key Benefits

1. **Full Reproducibility**: Every experiment is completely reproducible
2. **Artifact Storage**: Generated contracts are saved and downloadable
3. **Performance Tracking**: Monitor improvements over time
4. **Easy Comparison**: Side-by-side comparison of different approaches
5. **Search & Filter**: Find experiments by tags and parameters
6. **Version Control**: Track changes in your contract generation system

All your experiments are automatically saved to these locations with complete metadata and artifacts!