#!/usr/bin/env python3
"""
Example API client for the Lawyer Contract Creation System.
Demonstrates how to use the system programmatically.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional


class ContractSystemClient:
    """Client for interacting with the Contract Creation System API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def upload_skeleton(self, skeleton_path: str) -> str:
        """Upload a skeleton file and return the skeleton ID."""
        with open(skeleton_path, 'rb') as f:
            files = {'file': (Path(skeleton_path).name, f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
            response = self.session.post(f"{self.base_url}/upload/skeleton", files=files)
        
        response.raise_for_status()
        result = response.json()
        
        if not result.get('success'):
            raise Exception(f"Skeleton upload failed: {result}")
        
        print(f"✓ Skeleton uploaded successfully")
        print(f"  Skeleton ID: {result['skeleton_id']}")
        print(f"  Placeholders found: {', '.join(result['placeholders_found'])}")
        
        return result['skeleton_id']
    
    def upload_reference_contracts(self, reference_paths: list) -> str:
        """Upload reference contracts and return the reference ID."""
        files = []
        for path in reference_paths:
            with open(path, 'rb') as f:
                files.append(('files', (Path(path).name, f.read(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')))
        
        response = self.session.post(f"{self.base_url}/upload/references", files=files)
        response.raise_for_status()
        result = response.json()
        
        if not result.get('success'):
            raise Exception(f"Reference upload failed: {result}")
        
        print(f"✓ Reference contracts uploaded successfully")
        print(f"  Reference ID: {result['reference_id']}")
        print(f"  Contracts processed: {result['contracts_processed']}")
        
        return result['reference_id']
    
    def generate_contract(self, skeleton_id: str, contract_data: Dict[str, Any], 
                         checklist: Optional[list] = None, quality_threshold: float = 4.5,
                         reference_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a contract."""
        payload = {
            "contract_data": contract_data,
            "checklist": checklist or [],
            "quality_threshold": quality_threshold
        }
        
        params = {"skeleton_id": skeleton_id}
        if reference_id:
            params["reference_id"] = reference_id
        
        print(f"🔄 Starting contract generation...")
        print(f"  Quality threshold: {quality_threshold}")
        print(f"  Checklist items: {len(checklist) if checklist else 0}")
        
        response = self.session.post(
            f"{self.base_url}/contracts/generate",
            json=payload,
            params=params
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get('success'):
            print(f"✓ Contract generated successfully!")
            print(f"  Contract ID: {result['contract_id']}")
            print(f"  Quality Score: {result['quality_score']:.2f}")
            print(f"  Meets Threshold: {'Yes' if result['meets_threshold'] else 'No'}")
            print(f"  Generation Time: {result['generation_time']:.2f}s")
            print(f"  Iterations: {result['iterations']}")
            
            # Print quality gate results
            if result.get('quality_gates'):
                passed = sum(1 for gate in result['quality_gates'] if gate['status'] == 'passed')
                total = len(result['quality_gates'])
                print(f"  Quality Gates: {passed}/{total} passed")
        else:
            print(f"❌ Contract generation failed")
            if result.get('warnings'):
                print(f"  Warnings: {', '.join(result['warnings'])}")
        
        return result
    
    def download_contract(self, contract_id: str, output_path: str):
        """Download a generated contract."""
        response = self.session.get(f"{self.base_url}/contracts/download/{contract_id}")
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Contract downloaded to: {output_path}")
    
    def get_quality_report(self, contract_id: str) -> Dict[str, Any]:
        """Get detailed quality report for a contract."""
        response = self.session.get(f"{self.base_url}/contracts/quality-report/{contract_id}")
        response.raise_for_status()
        return response.json()
    
    def evaluate_contract_text(self, contract_text: str, reference_contracts: Optional[list] = None) -> Dict[str, Any]:
        """Evaluate existing contract text."""
        payload = {
            "contract_text": contract_text,
            "reference_contracts": reference_contracts or []
        }
        
        response = self.session.post(f"{self.base_url}/contracts/evaluate", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_generate_contracts(self, requests_list: list, parallel: bool = False) -> Dict[str, Any]:
        """Generate multiple contracts in batch."""
        payload = {
            "contracts": requests_list,
            "parallel_processing": parallel
        }
        
        print(f"🔄 Starting batch generation of {len(requests_list)} contracts...")
        
        response = self.session.post(f"{self.base_url}/contracts/batch-generate", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Batch generation completed!")
        print(f"  Total contracts: {result['total_contracts']}")
        print(f"  Successful: {result['successful_contracts']}")
        print(f"  Failed: {result['failed_contracts']}")
        print(f"  Processing time: {result['batch_processing_time']:.2f}s")
        
        return result


def example_service_agreement():
    """Example: Generate a service agreement contract."""
    print("=" * 60)
    print("Example 1: Service Agreement Contract Generation")
    print("=" * 60)
    
    client = ContractSystemClient()
    
    # Check system health
    try:
        health = client.health_check()
        print(f"✓ System is {health['status']}")
    except Exception as e:
        print(f"❌ System health check failed: {e}")
        return
    
    # Load example data
    with open('examples/example_contract_data.json', 'r') as f:
        example_data = json.load(f)
    
    service_agreement_data = example_data['service_agreement_example']
    
    # Upload skeleton
    try:
        skeleton_id = client.upload_skeleton('data/skeletons/service_agreement_skeleton.docx')
    except Exception as e:
        print(f"❌ Skeleton upload failed: {e}")
        return
    
    # Define checklist
    checklist = [
        "Include clear payment terms with specific amounts and schedules",
        "Add comprehensive confidentiality provisions",
        "Specify detailed termination conditions and procedures",
        "Define scope of work with measurable deliverables",
        "Include dispute resolution and governing law clauses"
    ]
    
    # Generate contract
    try:
        result = client.generate_contract(
            skeleton_id=skeleton_id,
            contract_data=service_agreement_data,
            checklist=checklist,
            quality_threshold=4.5
        )
        
        if result.get('success'):
            contract_id = result['contract_id']
            
            # Download the contract
            client.download_contract(contract_id, f"generated_service_agreement_{contract_id}.docx")
            
            # Get quality report
            quality_report = client.get_quality_report(contract_id)
            print(f"\n📊 Quality Report Summary:")
            print(f"  Overall Score: {quality_report['overall_quality_score']:.2f}")
            
            detailed_metrics = quality_report.get('detailed_metrics', {})
            for metric, score in detailed_metrics.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
            
            if quality_report.get('recommendations'):
                print(f"\n💡 Recommendations:")
                for rec in quality_report['recommendations']:
                    print(f"  - {rec}")
        
    except Exception as e:
        print(f"❌ Contract generation failed: {e}")


def example_nda_generation():
    """Example: Generate an NDA contract."""
    print("\n" + "=" * 60)
    print("Example 2: NDA Contract Generation")
    print("=" * 60)
    
    client = ContractSystemClient()
    
    # Load example data
    with open('examples/example_contract_data.json', 'r') as f:
        example_data = json.load(f)
    
    nda_data = example_data['nda_example']
    
    # Upload skeleton
    try:
        skeleton_id = client.upload_skeleton('data/skeletons/nda_skeleton.docx')
    except Exception as e:
        print(f"❌ Skeleton upload failed: {e}")
        return
    
    # NDA-specific checklist
    checklist = [
        "Define confidential information comprehensively",
        "Specify clear obligations for receiving party",
        "Include standard exceptions to confidentiality",
        "Set appropriate duration terms",
        "Add remedies for breach including injunctive relief"
    ]
    
    # Generate contract
    try:
        result = client.generate_contract(
            skeleton_id=skeleton_id,
            contract_data=nda_data,
            checklist=checklist,
            quality_threshold=4.0  # Slightly lower threshold for NDA
        )
        
        if result.get('success'):
            contract_id = result['contract_id']
            client.download_contract(contract_id, f"generated_nda_{contract_id}.docx")
            
    except Exception as e:
        print(f"❌ NDA generation failed: {e}")


def example_batch_generation():
    """Example: Batch contract generation."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Contract Generation")
    print("=" * 60)
    
    client = ContractSystemClient()
    
    # Create multiple contract requests
    batch_requests = []
    
    for i in range(3):
        contract_data = {
            "client_name": f"Client Company {i+1}",
            "provider_name": f"Service Provider {i+1}",
            "contract_value": f"${(i+1) * 25000}",
            "start_date": "2024-03-01",
            "service_description": f"Professional services for project {i+1}"
        }
        
        batch_requests.append({
            "contract_data": contract_data,
            "quality_threshold": 4.0,
            "checklist": ["Include payment terms", "Add confidentiality clause"]
        })
    
    try:
        result = client.batch_generate_contracts(batch_requests, parallel=False)
        
        # Download all successful contracts
        for i, contract_result in enumerate(result.get('results', [])):
            if contract_result.get('success'):
                contract_id = contract_result['contract_id']
                client.download_contract(contract_id, f"batch_contract_{i+1}_{contract_id}.docx")
                
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")


def example_contract_evaluation():
    """Example: Evaluate existing contract text."""
    print("\n" + "=" * 60)
    print("Example 4: Contract Text Evaluation")
    print("=" * 60)
    
    client = ContractSystemClient()
    
    # Sample contract text to evaluate
    contract_text = """
    SERVICE AGREEMENT
    
    This agreement is between Acme Corp and Provider LLC for consulting services.
    The total value is $50,000 payable monthly. Services include strategic planning
    and implementation support. The agreement is valid for one year starting
    January 1, 2024. Both parties agree to confidentiality of proprietary information.
    Either party may terminate with 30 days notice.
    """
    
    try:
        result = client.evaluate_contract_text(contract_text)
        
        print(f"✓ Contract evaluation completed")
        print(f"  Overall Assessment: {result.get('overall_assessment', 'N/A')}")
        
        if result.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
                
    except Exception as e:
        print(f"❌ Contract evaluation failed: {e}")


def main():
    """Run all examples."""
    print("🏛️ Lawyer Contract Creation System - API Client Examples")
    print("=" * 80)
    
    try:
        # Run examples
        example_service_agreement()
        example_nda_generation()
        example_batch_generation()
        example_contract_evaluation()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("📁 Generated contracts have been saved to the current directory")
        print("🔍 Check the quality reports for detailed evaluation metrics")
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()