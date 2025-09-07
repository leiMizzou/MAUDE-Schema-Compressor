#!/usr/bin/env python3
"""
Run complete MAUDE schema analysis using existing data files
This script demonstrates that the analysis can run without database connection
"""
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standalone_runner import StandaloneMAUDEAnalyzer

def main():
    """Run complete analysis"""
    print("=" * 60)
    print("MAUDE Schema Analysis - Complete Pipeline (No Database)")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Data directory - use local data directory
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    try:
        print(f"📁 Using data directory: {data_dir}")
        print(f"📊 Starting analysis...")
        
        # Initialize analyzer
        analyzer = StandaloneMAUDEAnalyzer(data_dir)
        
        # Run complete analysis
        results = analyzer.run_full_analysis()
        
        # Print summary
        clustering_only_count = len(results['clustering_only'])
        clustering_api_count = len(results['clustering_with_api'])
        total_experiments = clustering_only_count + clustering_api_count
        
        print("\n" + "=" * 60)
        print("📈 Analysis Results Summary:")
        print(f"  • Total experiments: {total_experiments}")
        print(f"  • Clustering only: {clustering_only_count}")
        print(f"  • Clustering + API similarity: {clustering_api_count}")
        
        # Show some sample results
        if clustering_only_count > 0:
            print(f"\n🔍 Sample Clustering Results:")
            for i, (key, result) in enumerate(list(results['clustering_only'].items())[:3]):
                metrics = result['metrics']
                info = result['experiment_info']
                clusters = result['clusters']
                print(f"  {i+1}. {info['clustering_method']} (param:{info['clustering_param']}, "
                      f"feature:{info['feature_extraction_method']})")
                print(f"     - Clusters: {len(clusters)}")
                if 'f1_score' in metrics:
                    print(f"     - F1 Score: {metrics['f1_score']:.3f}")
                if 'adjusted_rand_index' in metrics:
                    print(f"     - ARI: {metrics['adjusted_rand_index']:.3f}")
        
        if clustering_api_count > 0:
            print(f"\n🤖 Sample Clustering + API Results:")
            for i, (key, result) in enumerate(list(results['clustering_with_api'].items())[:3]):
                metrics = result['metrics']
                info = result['experiment_info']
                clusters = result['clusters']
                print(f"  {i+1}. {info['clustering_method']} + API (threshold:{info['similarity_threshold']})")
                print(f"     - Final clusters: {len(clusters)}")
                print(f"     - Pairs computed: {info['num_pairs_to_compute']}")
                if 'f1_score' in metrics:
                    print(f"     - F1 Score: {metrics['f1_score']:.3f}")
                if 'adjusted_rand_index' in metrics:
                    print(f"     - ARI: {metrics['adjusted_rand_index']:.3f}")
        
        # Cache statistics
        cache_stats = analyzer.similarity_calculator.get_cache_stats()
        print(f"\n💾 Cache Statistics:")
        print(f"  • Cached similarity scores: {cache_stats['cache_size']}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Results saved to: evaluation_results_standalone.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logging.exception("Detailed error information:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)