"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Test configuration
pytest_plugins = []


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_data"
        test_dir.mkdir(exist_ok=True)
        yield test_dir


@pytest.fixture
def sample_contract_data():
    """Fixture providing sample contract data."""
    return {
        "client_name": "Test Client Corporation",
        "client_address": "123 Test Street, Test City, TS 12345",
        "client_contact": "John Doe, CEO, john@testclient.com",
        "provider_name": "Professional Services LLC",
        "provider_address": "456 Service Avenue, Service City, SC 67890",
        "provider_contact": "Jane Smith, Partner, jane@proservices.com",
        "contract_type": "service_agreement",
        "contract_value": "$50,000",
        "start_date": "2024-03-01",
        "end_date": "2024-12-31",
        "service_description": "Professional consulting services including strategy development and implementation support",
        "payment_terms": "Monthly payments of $5,556, due within 30 days of invoice",
        "confidentiality_clause": "Both parties agree to maintain strict confidentiality of all proprietary information"
    }


@pytest.fixture
def sample_checklist():
    """Fixture providing sample contract checklist."""
    return [
        "Include clear payment terms",
        "Add confidentiality provisions",
        "Specify termination conditions", 
        "Define scope of work",
        "Include dispute resolution clause"
    ]


@pytest.fixture
def sample_reference_contracts():
    """Fixture providing sample reference contract texts."""
    return [
        """
        PROFESSIONAL SERVICES AGREEMENT
        
        This agreement is between Client Corp and Service Provider LLC.
        The provider will deliver consulting services including strategic planning.
        Total value is $60,000 with monthly payment schedule.
        Confidentiality provisions protect proprietary information.
        Either party may terminate with 30 days notice.
        """,
        """
        SERVICE CONTRACT
        
        Agreement between Business Inc and Consultant Group.
        Services include business analysis and process improvement.
        Contract value of $45,000 payable quarterly.
        Mutual confidentiality agreement protects sensitive data.
        Termination requires 60 days written notice.
        """
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('src.core.content_generator.OpenAI') as mock_client:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated contract content with professional legal language."
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 300
        
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('config.settings.settings') as mock_settings:
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.openai_model = "gpt-5"
        mock_settings.openai_temperature = 0.1
        mock_settings.min_bleu_score = 0.8
        mock_settings.min_rouge_average = 0.85
        mock_settings.min_meteor_score = 0.9
        mock_settings.min_comet_score = 0.8
        mock_settings.min_llm_judge_score = 4.5
        mock_settings.max_redundancy_score = 0.1
        mock_settings.min_completeness_score = 0.98
        mock_settings.max_regeneration_attempts = 3
        mock_settings.quality_gate_enabled = True
        mock_settings.data_dir = "test_data"
        mock_settings.skeletons_dir = "test_data/skeletons"
        mock_settings.generated_dir = "test_data/generated"
        mock_settings.references_dir = "test_data/references"
        yield mock_settings


@pytest.fixture
def create_test_docx():
    """Factory fixture for creating test .docx files."""
    created_files = []
    
    def _create_docx(content_lines, filename=None):
        from docx import Document
        
        if filename is None:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
            filename = tmp_file.name
            tmp_file.close()
        
        doc = Document()
        for line in content_lines:
            doc.add_paragraph(line)
        doc.save(filename)
        
        created_files.append(filename)
        return filename
    
    yield _create_docx
    
    # Cleanup
    for file_path in created_files:
        if os.path.exists(file_path):
            os.unlink(file_path)


@pytest.fixture
def sample_contract_skeleton():
    """Fixture providing sample contract skeleton content."""
    return [
        "SERVICE AGREEMENT",
        "",
        "This agreement is between {{client_name}} and {{provider_name}}.",
        "",
        "1. SERVICES",
        "{{service_description}}",
        "",
        "2. PAYMENT",
        "Total value: {{contract_value}}",
        "Payment terms: {{payment_terms}}",
        "",
        "3. DURATION",
        "Start: {{start_date}}",
        "End: {{end_date}}",
        "",
        "4. CONFIDENTIALITY",
        "{{confidentiality_clause}}",
        "",
        "Signatures:",
        "Client: _________________ Date: _______",
        "Provider: _______________ Date: _______"
    ]


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('src.utils.mlflow_tracker.mlflow') as mock_mlflow:
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.create_experiment.return_value = "test_experiment_id"
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value = Mock(info=Mock(run_id="test_run_id"))
        mock_mlflow.log_param.return_value = None
        mock_mlflow.log_metric.return_value = None
        mock_mlflow.log_artifact.return_value = None
        mock_mlflow.end_run.return_value = None
        yield mock_mlflow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_openai: mark test as requiring OpenAI API"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["slow", "comprehensive", "full"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark tests requiring OpenAI
        if any(keyword in item.nodeid for keyword in ["openai", "llm", "generation"]):
            item.add_marker(pytest.mark.requires_openai)


# Fixtures for specific test scenarios
@pytest.fixture
def high_quality_contract_text():
    """Fixture providing high-quality contract text for evaluation."""
    return """
    PROFESSIONAL SERVICES AGREEMENT
    
    This Professional Services Agreement ("Agreement") is entered into on March 1, 2024,
    between TechCorp Solutions Inc., a Delaware corporation ("Client"), and Expert Legal
    Services LLC, a New York limited liability company ("Provider").
    
    1. SCOPE OF SERVICES
    Provider agrees to provide comprehensive legal consultation services including:
    - Contract review and analysis for all commercial agreements exceeding $25,000
    - Regulatory compliance consulting for technology sector requirements
    - Risk assessment and mitigation strategies for business partnerships
    - Monthly legal advisory sessions with executive leadership team
    
    2. COMPENSATION AND PAYMENT TERMS
    Total contract value is Seventy-Five Thousand Dollars ($75,000) annually.
    Payment shall be made in quarterly installments of $18,750 each, due within
    thirty (30) days of Provider's invoice. Late payments shall incur a service
    charge of 1.5% per month.
    
    3. CONFIDENTIALITY
    Both parties acknowledge that confidential information may be disclosed during
    performance of this Agreement. All proprietary information shall remain strictly
    confidential and shall not be disclosed to third parties without prior written
    consent. This obligation survives termination for five (5) years.
    
    4. TERM AND TERMINATION
    This Agreement commences on March 1, 2024, and continues until February 28, 2025.
    Either party may terminate with sixty (60) days written notice. Upon termination,
    all confidential information must be returned and outstanding fees become due.
    
    5. GOVERNING LAW
    This Agreement shall be governed by the laws of New York State. Disputes shall
    be resolved through binding arbitration under AAA Commercial Rules.
    """


@pytest.fixture
def poor_quality_contract_text():
    """Fixture providing poor-quality contract text for evaluation."""
    return """
    Contract thing
    
    This is a contract. The client will pay money. The provider will do work.
    Money is good. Work is also good. Both parties agree to do things.
    
    Payment: Some money at some time
    Work: Some work when needed
    Time: Starts now, ends later
    
    If problems happen, fix them. If can't fix, then stop contract.
    This contract is legal and binding and stuff.
    
    Sign here: _______________
    """