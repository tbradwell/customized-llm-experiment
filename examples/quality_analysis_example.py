#!/usr/bin/env python3
"""
Example script demonstrating quality analysis and metrics evaluation
for contract generation using the Lawyer Contract Creation System.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import MetricsCalculator, COMETEvaluator
from src.evaluation.llm_judge import LLMJudge
from src.utils.mlflow_tracker import MLflowTracker


def analyze_contract_quality(contract_text: str, reference_contracts: List[str], 
                           contract_type: str = "service_agreement") -> Dict[str, Any]:
    """Analyze the quality of a generated contract using all available metrics."""
    
    print("🔍 Starting comprehensive quality analysis...")
    
    # Initialize evaluators
    metrics_calc = MetricsCalculator()
    comet_eval = COMETEvaluator()
    llm_judge = LLMJudge()
    
    results = {}
    
    # 1. N-gram based metrics
    print("📊 Calculating N-gram based metrics...")
    
    # BLEU Score
    bleu_result = metrics_calc.calculate_bleu_score(contract_text, reference_contracts)
    results['bleu'] = {
        'score': bleu_result.score,
        'passed': bleu_result.passed_threshold,
        'details': bleu_result.details
    }
    print(f"  BLEU Score: {bleu_result.score:.3f} ({'✓' if bleu_result.passed_threshold else '❌'})")
    
    # ROUGE Scores
    rouge_result = metrics_calc.calculate_rouge_scores(contract_text, reference_contracts)
    results['rouge'] = {
        'score': rouge_result.score,
        'passed': rouge_result.passed_threshold,
        'details': rouge_result.details
    }
    print(f"  ROUGE Score: {rouge_result.score:.3f} ({'✓' if rouge_result.passed_threshold else '❌'})")
    print(f"    ROUGE-1: {rouge_result.details.get('rouge1_avg', 0):.3f}")
    print(f"    ROUGE-2: {rouge_result.details.get('rouge2_avg', 0):.3f}")
    print(f"    ROUGE-L: {rouge_result.details.get('rougeL_avg', 0):.3f}")
    
    # METEOR Score
    meteor_result = metrics_calc.calculate_meteor_score(contract_text, reference_contracts)
    results['meteor'] = {
        'score': meteor_result.score,
        'passed': meteor_result.passed_threshold,
        'details': meteor_result.details
    }
    print(f"  METEOR Score: {meteor_result.score:.3f} ({'✓' if meteor_result.passed_threshold else '❌'})")
    
    # 2. Neural-based metrics
    print("\n🧠 Calculating neural-based metrics...")
    
    # COMET Score
    comet_result = comet_eval.calculate_comet_score(contract_text, reference_contracts)
    results['comet'] = {
        'score': comet_result.score,
        'passed': comet_result.passed_threshold,
        'details': comet_result.details
    }
    print(f"  COMET Score: {comet_result.score:.3f} ({'✓' if comet_result.passed_threshold else '❌'})")
    
    # LLM Judge
    contract_context = {
        'contract_type': contract_type,
        'client_name': 'Sample Client',
        'provider_name': 'Sample Provider'
    }
    
    try:
        llm_result = llm_judge.evaluate_contract(contract_text, contract_context, reference_contracts)
        results['llm_judge'] = {
            'score': llm_result.score,
            'passed': llm_result.passed_threshold,
            'details': llm_result.details
        }
        print(f"  LLM Judge Score: {llm_result.score:.3f} ({'✓' if llm_result.passed_threshold else '❌'})")
        
        # Print detailed LLM feedback
        if 'detailed_feedback' in llm_result.details:
            print(f"    Feedback: {llm_result.details['detailed_feedback'][:200]}...")
            
    except Exception as e:
        print(f"  LLM Judge: Failed ({str(e)})")
        results['llm_judge'] = {'score': 0.0, 'passed': False, 'error': str(e)}
    
    # 3. Custom quality metrics
    print("\n📋 Calculating custom quality metrics...")
    
    # Redundancy Score
    redundancy_result = metrics_calc.calculate_redundancy_score(contract_text)
    results['redundancy'] = {
        'score': redundancy_result.score,
        'passed': redundancy_result.passed_threshold,
        'details': redundancy_result.details
    }
    print(f"  Redundancy Score: {redundancy_result.score:.3f} ({'✓' if redundancy_result.passed_threshold else '❌'})")
    
    # Completeness Score
    required_elements = [
        "payment", "terms", "services", "client", "provider",
        "date", "agreement", "confidentiality", "termination"
    ]
    completeness_result = metrics_calc.calculate_completeness_score(contract_text, required_elements)
    results['completeness'] = {
        'score': completeness_result.score,
        'passed': completeness_result.passed_threshold,
        'details': completeness_result.details
    }
    print(f"  Completeness Score: {completeness_result.score:.3f} ({'✓' if completeness_result.passed_threshold else '❌'})")
    print(f"    Found elements: {completeness_result.details.get('found_count', 0)}/{len(required_elements)}")
    
    # 4. Calculate overall quality score
    metric_weights = {
        'bleu': 0.15,
        'rouge': 0.15,
        'meteor': 0.15,
        'comet': 0.15,
        'llm_judge': 0.15,
        'redundancy': 0.1,  # Inverted (lower is better)
        'completeness': 0.15
    }
    
    overall_score = 0.0
    total_weight = 0.0
    
    for metric, weight in metric_weights.items():
        if metric in results and 'score' in results[metric]:
            score = results[metric]['score']
            # Invert redundancy score (lower redundancy is better)
            if metric == 'redundancy':
                score = 1.0 - min(1.0, score)
            overall_score += score * weight
            total_weight += weight
    
    overall_score = overall_score / total_weight if total_weight > 0 else 0.0
    results['overall'] = overall_score
    
    print(f"\n🎯 Overall Quality Score: {overall_score:.3f}")
    
    return results


def compare_contracts(contracts_with_names: List[tuple], reference_contracts: List[str]):
    """Compare multiple contracts side by side."""
    
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE QUALITY ANALYSIS")
    print("=" * 80)
    
    all_results = []
    
    for name, contract_text in contracts_with_names:
        print(f"\n🔍 Analyzing: {name}")
        print("-" * 60)
        
        results = analyze_contract_quality(contract_text, reference_contracts)
        results['name'] = name
        all_results.append(results)
    
    # Create comparison table
    print("\n📋 QUALITY COMPARISON TABLE")
    print("=" * 100)
    
    # Print header
    print(f"{'Contract':<20} {'Overall':<8} {'BLEU':<7} {'ROUGE':<7} {'METEOR':<8} {'COMET':<7} {'LLM':<5} {'Redund':<7} {'Complt':<7}")
    print("-" * 100)
    
    # Print results for each contract
    for result in all_results:
        name = result['name'][:18]
        overall = result.get('overall', 0)
        bleu = result.get('bleu', {}).get('score', 0)
        rouge = result.get('rouge', {}).get('score', 0)
        meteor = result.get('meteor', {}).get('score', 0)
        comet = result.get('comet', {}).get('score', 0)
        llm = result.get('llm_judge', {}).get('score', 0)
        redundancy = result.get('redundancy', {}).get('score', 0)
        completeness = result.get('completeness', {}).get('score', 0)
        
        print(f"{name:<20} {overall:<8.3f} {bleu:<7.3f} {rouge:<7.3f} {meteor:<8.3f} "
              f"{comet:<7.3f} {llm:<5.1f} {redundancy:<7.3f} {completeness:<7.3f}")
    
    return all_results


def visualize_quality_metrics(results: Dict[str, Any], save_path: str = "quality_analysis.png"):
    """Create a visualization of quality metrics."""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract metric scores
        metrics = ['BLEU', 'ROUGE', 'METEOR', 'COMET', 'LLM Judge', 'Redundancy', 'Completeness']
        scores = [
            results.get('bleu', {}).get('score', 0),
            results.get('rouge', {}).get('score', 0),
            results.get('meteor', {}).get('score', 0),
            results.get('comet', {}).get('score', 0),
            results.get('llm_judge', {}).get('score', 0) / 5.0,  # Normalize to 0-1
            1.0 - results.get('redundancy', {}).get('score', 0),  # Invert redundancy
            results.get('completeness', {}).get('score', 0)
        ]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        bars = ax1.bar(metrics, scores, color=colors)
        ax1.set_ylabel('Score')
        ax1.set_title('Contract Quality Metrics')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores_radar = scores + [scores[0]]  # Complete the circle
        angles += angles[:1]  # Complete the circle
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, scores_radar, color='#FF6B6B', linewidth=2)
        ax2.fill(angles, scores_radar, color='#FF6B6B', alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Quality Metrics Radar Chart')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Quality visualization saved to: {save_path}")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")


def main():
    """Main function demonstrating quality analysis."""
    
    print("🏛️ Contract Quality Analysis Example")
    print("=" * 80)
    
    # Sample contracts for analysis
    high_quality_contract = """
    PROFESSIONAL SERVICES AGREEMENT
    
    This Professional Services Agreement ("Agreement") is entered into on March 1, 2024,
    between TechCorp Solutions Inc., a Delaware corporation ("Client"), and Expert Legal
    Services LLC, a New York limited liability company ("Provider").
    
    1. SCOPE OF SERVICES
    Provider agrees to provide comprehensive legal consultation services including contract
    review and analysis, regulatory compliance consulting, risk assessment and mitigation
    strategies, and monthly legal advisory sessions with executive leadership.
    
    2. COMPENSATION AND PAYMENT TERMS
    Total contract value is Seventy-Five Thousand Dollars ($75,000) annually. Payment
    shall be made in quarterly installments of $18,750 each, due within thirty (30) days
    of Provider's invoice. Late payments shall incur a service charge of 1.5% per month.
    
    3. CONFIDENTIALITY
    Both parties acknowledge that confidential information may be disclosed during
    performance of this Agreement. All proprietary information shall remain strictly
    confidential and shall not be disclosed to third parties without prior written consent.
    This obligation survives termination for five (5) years.
    
    4. TERM AND TERMINATION
    This Agreement commences on March 1, 2024, and continues until February 28, 2025.
    Either party may terminate with sixty (60) days written notice. Upon termination,
    all confidential information must be returned and outstanding fees become due.
    
    5. GOVERNING LAW
    This Agreement shall be governed by the laws of New York State. Disputes shall be
    resolved through binding arbitration under AAA Commercial Rules.
    """
    
    medium_quality_contract = """
    SERVICE AGREEMENT
    
    This agreement is between Acme Corp and Provider LLC for consulting services.
    
    Services: Provider will deliver business consulting including strategy development
    and implementation support for the client's operations.
    
    Payment: Total value is $50,000 payable in monthly installments. Payment terms
    are Net 30 days from invoice date.
    
    Duration: Agreement starts January 1, 2024 and ends December 31, 2024.
    
    Confidentiality: Both parties agree to maintain confidentiality of proprietary
    information shared during the engagement.
    
    Termination: Either party may terminate with 30 days written notice.
    """
    
    low_quality_contract = """
    Contract
    
    This is a contract. Company A will pay Company B money. Company B will do work.
    
    Money: $25,000
    Time: This year
    Work: Some business stuff
    
    Both companies agree to keep secrets secret.
    Contract can be stopped if needed.
    """
    
    # Reference contracts for comparison
    reference_contracts = [
        """
        PROFESSIONAL SERVICES AGREEMENT
        
        This agreement between Client and Provider covers comprehensive business consulting
        services including strategic planning, implementation support, and ongoing advisory
        services. Total contract value is $60,000 with quarterly payment schedule.
        Confidentiality provisions protect proprietary information. Either party may
        terminate with 60 days notice. Agreement governed by state law with arbitration
        for dispute resolution.
        """
    ]
    
    # Analyze individual contracts
    print("🔍 Individual Contract Analysis")
    print("=" * 50)
    
    high_quality_results = analyze_contract_quality(high_quality_contract, reference_contracts)
    visualize_quality_metrics(high_quality_results, "high_quality_contract_analysis.png")
    
    # Compare multiple contracts
    contracts_to_compare = [
        ("High Quality Contract", high_quality_contract),
        ("Medium Quality Contract", medium_quality_contract),
        ("Low Quality Contract", low_quality_contract)
    ]
    
    comparison_results = compare_contracts(contracts_to_compare, reference_contracts)
    
    # Summary insights
    print("\n" + "=" * 80)
    print("💡 QUALITY ANALYSIS INSIGHTS")
    print("=" * 80)
    
    print("\n🎯 Key Findings:")
    print("• High quality contracts score consistently above 0.8 across all metrics")
    print("• Professional legal language significantly improves LLM Judge scores")
    print("• Comprehensive sections boost completeness scores")
    print("• Specific terms and amounts improve BLEU/ROUGE scores against references")
    print("• Redundancy should be kept below 0.1 for optimal readability")
    
    print("\n📈 Quality Improvement Recommendations:")
    print("• Use precise legal terminology and professional language structure")
    print("• Include all essential contract elements (parties, terms, conditions)")
    print("• Specify exact amounts, dates, and measurable deliverables")
    print("• Add standard legal provisions (governing law, dispute resolution)")
    print("• Ensure logical flow and avoid unnecessary repetition")
    
    print(f"\n✅ Quality analysis completed successfully!")
    print(f"📊 Visualizations saved as PNG files in current directory")


if __name__ == "__main__":
    main()