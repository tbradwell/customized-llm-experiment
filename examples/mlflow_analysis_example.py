#!/usr/bin/env python3
"""
Example script demonstrating MLflow experiment tracking and analysis
for the Lawyer Contract Creation System.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.mlflow_tracker import MLflowTracker


def analyze_experiment_performance(tracker: MLflowTracker, days_back: int = 30):
    """Analyze experiment performance over time."""
    
    print("📊 MLflow Experiment Performance Analysis")
    print("=" * 60)
    
    try:
        # Get recent runs
        runs_df = tracker.get_experiment_runs(max_results=100)
        
        if runs_df.empty:
            print("❌ No experiment runs found")
            return
        
        print(f"📈 Found {len(runs_df)} experiment runs")
        
        # Filter runs from last N days
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_runs = runs_df[runs_df['start_time'] >= cutoff_date.timestamp() * 1000]
        
        print(f"🗓️ Analyzing {len(recent_runs)} runs from last {days_back} days")
        
        # Quality metrics analysis
        quality_metrics = [
            'metrics.quality_bleu', 'metrics.quality_rouge', 'metrics.quality_meteor',
            'metrics.quality_comet', 'metrics.quality_llm_judge', 'metrics.quality_redundancy',
            'metrics.quality_completeness', 'metrics.overall_quality_score'
        ]
        
        print("\n📊 Quality Metrics Summary:")
        for metric in quality_metrics:
            if metric in recent_runs.columns:
                values = recent_runs[metric].dropna()
                if not values.empty:
                    print(f"  {metric.replace('metrics.quality_', '').replace('metrics.', '').title()}: "
                          f"μ={values.mean():.3f}, σ={values.std():.3f}, "
                          f"min={values.min():.3f}, max={values.max():.3f}")
        
        # Performance metrics
        print("\n⚡ Performance Metrics:")
        
        if 'metrics.generation_time_seconds' in recent_runs.columns:
            gen_times = recent_runs['metrics.generation_time_seconds'].dropna()
            if not gen_times.empty:
                print(f"  Generation Time: μ={gen_times.mean():.2f}s, "
                      f"σ={gen_times.std():.2f}s, max={gen_times.max():.2f}s")
        
        if 'metrics.quality_iterations' in recent_runs.columns:
            iterations = recent_runs['metrics.quality_iterations'].dropna()
            if not iterations.empty:
                print(f"  Quality Iterations: μ={iterations.mean():.1f}, "
                      f"max={int(iterations.max())}")
        
        # Quality gate analysis
        if 'metrics.quality_gates_pass_rate' in recent_runs.columns:
            pass_rates = recent_runs['metrics.quality_gates_pass_rate'].dropna()
            if not pass_rates.empty:
                success_rate = (pass_rates == 1.0).sum() / len(pass_rates) * 100
                print(f"  Quality Gate Success Rate: {success_rate:.1f}%")
        
        # Contract type analysis
        if 'params.contract_type' in recent_runs.columns:
            contract_types = recent_runs['params.contract_type'].value_counts()
            print(f"\n📋 Contract Types:")
            for contract_type, count in contract_types.items():
                print(f"  {contract_type}: {count} contracts")
        
        # Model performance by type
        print("\n🤖 Model Performance Analysis:")
        
        if 'params.openai_model' in recent_runs.columns:
            models = recent_runs['params.openai_model'].value_counts()
            print(f"  Models used: {', '.join(models.index.tolist())}")
            
            for model in models.index:
                model_runs = recent_runs[recent_runs['params.openai_model'] == model]
                if 'metrics.overall_quality_score' in model_runs.columns:
                    scores = model_runs['metrics.overall_quality_score'].dropna()
                    if not scores.empty:
                        print(f"  {model}: μ={scores.mean():.3f} (n={len(scores)})")
        
        return recent_runs
        
    except Exception as e:
        print(f"❌ Error analyzing experiments: {e}")
        return None


def find_best_performing_runs(tracker: MLflowTracker, metric: str = "overall_quality_score", top_k: int = 5):
    """Find the best performing runs based on a specific metric."""
    
    print(f"\n🏆 Top {top_k} Best Performing Runs (by {metric})")
    print("=" * 60)
    
    try:
        best_runs = tracker.get_best_runs_by_metric(f"metrics.{metric}", top_k)
        
        if best_runs.empty:
            print("❌ No runs found with the specified metric")
            return
        
        print(f"{'Rank':<5} {'Run ID':<12} {'Score':<8} {'Contract Type':<15} {'Generation Time':<12}")
        print("-" * 65)
        
        for i, (_, run) in enumerate(best_runs.iterrows(), 1):
            run_id = run.get('run_id', 'N/A')[:10]
            score = run.get(f'metrics.{metric}', 0)
            contract_type = run.get('params.contract_type', 'Unknown')
            gen_time = run.get('metrics.generation_time_seconds', 0)
            
            print(f"{i:<5} {run_id:<12} {score:<8.3f} {contract_type:<15} {gen_time:<8.2f}s")
        
        # Analyze patterns in best runs
        print(f"\n🔍 Analysis of Top {top_k} Runs:")
        
        # Most common contract type
        if 'params.contract_type' in best_runs.columns:
            top_type = best_runs['params.contract_type'].mode()
            if not top_type.empty:
                print(f"  Most successful contract type: {top_type.iloc[0]}")
        
        # Average metrics for top runs
        quality_columns = [col for col in best_runs.columns if col.startswith('metrics.quality_')]
        if quality_columns:
            print(f"  Average quality metrics for top runs:")
            for col in quality_columns:
                values = best_runs[col].dropna()
                if not values.empty:
                    metric_name = col.replace('metrics.quality_', '').replace('metrics.', '')
                    print(f"    {metric_name}: {values.mean():.3f}")
        
        return best_runs
        
    except Exception as e:
        print(f"❌ Error finding best runs: {e}")
        return None


def compare_experiment_runs(tracker: MLflowTracker, run_ids: List[str]):
    """Compare specific experiment runs."""
    
    print(f"\n🔄 Comparing {len(run_ids)} Experiment Runs")
    print("=" * 60)
    
    try:
        comparison = tracker.compare_runs(run_ids)
        
        if not comparison.get('runs'):
            print("❌ No run data found for comparison")
            return
        
        # Display run information
        print(f"{'Metric':<25} " + " ".join([f"Run {i+1:<10}" for i in range(len(run_ids))]))
        print("-" * (25 + 12 * len(run_ids)))
        
        # Compare key metrics
        metrics_comparison = comparison.get('metrics_comparison', {})
        important_metrics = [
            'overall_quality_score', 'quality_bleu', 'quality_rouge', 
            'quality_meteor', 'quality_llm_judge', 'generation_time_seconds'
        ]
        
        for metric in important_metrics:
            if metric in metrics_comparison:
                values = metrics_comparison[metric]
                formatted_values = []
                for val in values:
                    if val is not None:
                        formatted_values.append(f"{val:<10.3f}")
                    else:
                        formatted_values.append(f"{'N/A':<10}")
                
                print(f"{metric:<25} " + " ".join(formatted_values))
        
        # Find the best run
        overall_scores = metrics_comparison.get('overall_quality_score', [])
        if overall_scores and any(score is not None for score in overall_scores):
            best_idx = max(range(len(overall_scores)), 
                          key=lambda i: overall_scores[i] if overall_scores[i] is not None else -1)
            print(f"\n🎯 Best performing run: Run {best_idx + 1} (ID: {run_ids[best_idx][:10]})")
        
        return comparison
        
    except Exception as e:
        print(f"❌ Error comparing runs: {e}")
        return None


def analyze_quality_trends(tracker: MLflowTracker, days_back: int = 30):
    """Analyze quality trends over time."""
    
    print(f"\n📈 Quality Trends Analysis (Last {days_back} days)")
    print("=" * 60)
    
    try:
        runs_df = tracker.get_experiment_runs(max_results=200)
        
        if runs_df.empty:
            print("❌ No experiment data available")
            return
        
        # Filter recent runs and sort by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_runs = runs_df[runs_df['start_time'] >= cutoff_date.timestamp() * 1000].copy()
        recent_runs = recent_runs.sort_values('start_time')
        
        if recent_runs.empty:
            print(f"❌ No runs found in the last {days_back} days")
            return
        
        # Calculate daily averages
        recent_runs['date'] = pd.to_datetime(recent_runs['start_time'], unit='ms').dt.date
        
        quality_metrics = ['overall_quality_score', 'quality_bleu', 'quality_rouge', 'quality_meteor']
        
        print("📊 Daily Quality Trends:")
        
        daily_stats = recent_runs.groupby('date').agg({
            f'metrics.{metric}': ['mean', 'count'] for metric in quality_metrics if f'metrics.{metric}' in recent_runs.columns
        })
        
        # Display trends
        for date_val in daily_stats.index[-7:]:  # Last 7 days
            date_data = daily_stats.loc[date_val]
            print(f"\n  {date_val}:")
            
            for metric in quality_metrics:
                col_name = f'metrics.{metric}'
                if (col_name, 'mean') in date_data.index:
                    mean_score = date_data[(col_name, 'mean')]
                    count = date_data[(col_name, 'count')]
                    if not pd.isna(mean_score):
                        print(f"    {metric}: {mean_score:.3f} (n={int(count)})")
        
        # Calculate trend direction
        if 'metrics.overall_quality_score' in recent_runs.columns:
            scores = recent_runs['metrics.overall_quality_score'].dropna()
            if len(scores) >= 2:
                # Simple linear trend
                x = np.arange(len(scores))
                slope = np.corrcoef(x, scores)[0, 1]
                
                if slope > 0.1:
                    trend = "📈 Improving"
                elif slope < -0.1:
                    trend = "📉 Declining"
                else:
                    trend = "➡️ Stable"
                
                print(f"\n🎯 Overall Quality Trend: {trend}")
        
        return recent_runs
        
    except Exception as e:
        print(f"❌ Error analyzing trends: {e}")
        return None


def generate_experiment_report(tracker: MLflowTracker):
    """Generate a comprehensive experiment report."""
    
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE EXPERIMENT REPORT")
    print("=" * 80)
    
    # System overview
    print("\n🖥️ System Overview:")
    try:
        all_runs = tracker.get_experiment_runs(max_results=1000)
        total_runs = len(all_runs)
        
        if total_runs > 0:
            # Date range
            start_date = pd.to_datetime(all_runs['start_time'].min(), unit='ms')
            end_date = pd.to_datetime(all_runs['start_time'].max(), unit='ms')
            
            print(f"  Total Experiment Runs: {total_runs}")
            print(f"  Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"  Duration: {(end_date - start_date).days} days")
            
            # Success rate
            if 'metrics.overall_quality_score' in all_runs.columns:
                quality_scores = all_runs['metrics.overall_quality_score'].dropna()
                high_quality = (quality_scores >= 4.0).sum()
                success_rate = high_quality / len(quality_scores) * 100
                print(f"  High Quality Rate (≥4.0): {success_rate:.1f}%")
        
    except Exception as e:
        print(f"  Error generating overview: {e}")
    
    # Recent performance
    print("\n⚡ Recent Performance (Last 7 days):")
    recent_analysis = analyze_experiment_performance(tracker, days_back=7)
    
    # Best performers
    print("\n🏆 Best Performers:")
    best_runs = find_best_performing_runs(tracker, top_k=3)
    
    # Quality trends
    print("\n📈 Quality Trends:")
    trends = analyze_quality_trends(tracker, days_back=14)
    
    # Recommendations
    print("\n💡 Recommendations:")
    try:
        if recent_analysis is not None and not recent_analysis.empty:
            # Generate recommendations based on data
            recommendations = []
            
            if 'metrics.overall_quality_score' in recent_analysis.columns:
                avg_quality = recent_analysis['metrics.overall_quality_score'].mean()
                if avg_quality < 4.0:
                    recommendations.append("• Consider adjusting quality thresholds or improving prompts")
                else:
                    recommendations.append("• Maintain current quality standards - performing well")
            
            if 'metrics.generation_time_seconds' in recent_analysis.columns:
                avg_time = recent_analysis['metrics.generation_time_seconds'].mean()
                if avg_time > 10.0:
                    recommendations.append("• Optimize generation process to reduce average time")
            
            if 'metrics.quality_iterations' in recent_analysis.columns:
                avg_iterations = recent_analysis['metrics.quality_iterations'].mean()
                if avg_iterations > 2.0:
                    recommendations.append("• Review quality gates - high iteration count may indicate issues")
            
            if recommendations:
                for rec in recommendations:
                    print(f"  {rec}")
            else:
                print("  • System is performing within expected parameters")
        else:
            print("  • No recent data available for recommendations")
            
    except Exception as e:
        print(f"  Error generating recommendations: {e}")
    
    print("\n✅ Experiment report completed")


def main():
    """Main function demonstrating MLflow analysis capabilities."""
    
    print("🔬 MLflow Experiment Tracking and Analysis")
    print("=" * 80)
    
    # Initialize MLflow tracker
    try:
        tracker = MLflowTracker()
        print("✅ MLflow tracker initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MLflow tracker: {e}")
        print("💡 Make sure MLflow is properly configured and running")
        return
    
    # Run various analysis functions
    try:
        # Basic performance analysis
        analyze_experiment_performance(tracker, days_back=30)
        
        # Find best performing runs
        find_best_performing_runs(tracker, top_k=5)
        
        # Analyze quality trends
        analyze_quality_trends(tracker, days_back=14)
        
        # Generate comprehensive report
        generate_experiment_report(tracker)
        
        print("\n" + "=" * 80)
        print("🎯 Key Insights for Quality Improvement:")
        print("=" * 80)
        print("• Monitor overall quality scores - aim for consistent scores above 4.0")
        print("• Track generation time - optimize if consistently above 10 seconds")
        print("• Watch quality gate pass rates - investigate if below 80%")
        print("• Compare model performance - use best performing configurations")
        print("• Analyze contract type patterns - focus on successful patterns")
        print("• Review iteration counts - high counts may indicate prompt issues")
        
        print("\n📊 MLflow Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()