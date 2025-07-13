"""
Standalone runner for MAUDE Schema Compressor
Runs analysis using existing data files without database connection
"""
import os
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from database import TableDataManager, DatabaseManager
from features import FeatureProcessor
from clustering import ClusteringManager
from similarity import SimilarityCalculator
from evaluation import Evaluator
from table_merger import TableSchemaMerger

class StandaloneMAUDEAnalyzer:
    """Standalone analyzer that works with existing data files"""
    
    def __init__(self, data_directory: str = None, update_cache: bool = True):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Store data directory separately from output directory
        self.data_directory = data_directory or './data'
        self.update_cache = update_cache
        self.logger.info(f"Using data directory: {self.data_directory}")
        self.logger.info(f"Using output directory: {config.paths.output_dir}")
        self.logger.info(f"Cache update mode: {'enabled' if update_cache else 'disabled'}")
        
        # Initialize components (no database connection needed)
        self.table_manager = TableDataManager(None)  # No DB manager needed
        self.feature_processor = FeatureProcessor()
        self.clustering_manager = ClusteringManager()
        self.similarity_calculator = SimilarityCalculator(update_cache=update_cache)
        self.evaluator = Evaluator()
        self.table_merger = TableSchemaMerger()
        
        self.logger.info("Standalone MAUDE Analyzer initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('standalone_analysis.log')
            ]
        )
    
    def verify_data_availability(self) -> bool:
        """
        Verify that required data files are available
        """
        self.logger.info(f"Checking data availability in: {self.data_directory}")
        
        if not os.path.exists(self.data_directory):
            self.logger.error(f"Data directory does not exist: {self.data_directory}")
            return False
        
        # Temporarily override output_dir for loading data
        original_output_dir = config.paths.output_dir
        config.paths.output_dir = self.data_directory
        
        # Load existing data
        try:
            table_data = self.table_manager.load_all_table_data()
            if not table_data:
                self.logger.error("No table data found in the directory")
                return False
            
            self.logger.info(f"Found data for {len(table_data)} tables")
            
            # Check for essential files
            essential_files = [
                config.paths.cache_file,
                config.paths.manual_grouping_file
            ]
            
            missing_files = []
            for file_path in essential_files:
                full_path = os.path.join(self.data_directory, os.path.basename(file_path))
                if not os.path.exists(full_path):
                    # Try original location
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)
            
            if missing_files:
                self.logger.warning(f"Missing optional files: {missing_files}")
                self.logger.info("Will continue without these files (some features may be limited)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying data: {e}")
            return False
        finally:
            # Restore original output directory
            config.paths.output_dir = original_output_dir
    
    def adjust_file_paths(self):
        """
        Adjust file paths to use the data directory for input and output directory for results
        """
        data_dir = self.data_directory
        output_dir = config.paths.output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Update cache file path (read from data directory)
        cache_file = os.path.join(data_dir, 'similarity_cache.json')
        if os.path.exists(cache_file):
            config.paths.cache_file = cache_file
            # Reinitialize similarity calculator with correct cache path
            self.similarity_calculator = SimilarityCalculator()
            self.logger.info(f"Using cache file: {cache_file}")
        
        # Update manual grouping file path - check multiple locations
        manual_grouping_locations = [
            os.path.join(data_dir, 'manual_grouping.json'),  # In data directory
            os.path.join(output_dir, 'manual_grouping.json'),  # In output directory
            'manual_grouping.json'  # Current directory
        ]
        
        manual_grouping_found = False
        for manual_grouping_file in manual_grouping_locations:
            if os.path.exists(manual_grouping_file):
                config.paths.manual_grouping_file = manual_grouping_file
                self.logger.info(f"Using manual grouping file: {manual_grouping_file}")
                manual_grouping_found = True
                break
        
        if not manual_grouping_found:
            self.logger.warning("Manual grouping file not found in any expected location")
        
        # Update output file paths (save to output directory)
        config.paths.evaluation_results_file = os.path.join(output_dir, 'evaluation_results_standalone.csv')
        
        self.logger.info(f"File paths adjusted - Data: {data_dir}, Output: {output_dir}")
    
    def run_clustering_analysis(self) -> dict:
        """
        Run clustering analysis using existing data
        """
        self.logger.info("Starting clustering analysis")
        
        # Temporarily set data directory for loading
        original_output_dir = config.paths.output_dir
        config.paths.output_dir = self.data_directory
        
        try:
            # Load table data and generate descriptions
            table_data = self.table_manager.load_all_table_data()
            descriptions = self.table_manager.get_all_table_descriptions()
        finally:
            # Restore output directory
            config.paths.output_dir = original_output_dir
        
        # Prepare table structures
        table_structures = {}
        for table_name, data in table_data.items():
            table_structures[table_name] = data.get('structure', [])
        
        results = {'clustering_only': {}, 'clustering_with_api': {}}
        
        # Run experiments for each feature extraction method
        for feature_method in config.processing.feature_extraction_methods:
            self.logger.info(f"Processing with feature method: {feature_method}")
            
            try:
                # Extract features
                features, table_names = self.feature_processor.process_features(
                    descriptions, table_structures, feature_method
                )
                
                # Run clustering experiments
                for method, method_config in config.clustering.clustering_methods.items():
                    for param in method_config['params']:
                        self.logger.info(f"Running clustering: {method} with param {param}")
                        
                        try:
                            # Perform clustering
                            clusters = self.clustering_manager.cluster_tables(
                                features, table_names, method, param
                            )
                            
                            # Evaluate clustering-only results
                            experiment_info = {
                                'experiment_type': 'clustering_only',
                                'clustering_method': method,
                                'clustering_param': param,
                                'feature_extraction_method': feature_method,
                                'similarity_threshold': None,
                                'num_pairs_to_compute': 0
                            }
                            
                            metrics = self.evaluator.evaluate_clustering_result(
                                clusters, table_names, experiment_info
                            )
                            
                            key = f"{method}_{param}_{feature_method}"
                            results['clustering_only'][key] = {
                                'clusters': clusters,
                                'metrics': metrics,
                                'experiment_info': experiment_info
                            }
                            
                            # Run API similarity experiments if enabled and cache exists
                            if config.processing.enable_similarity_calculation:
                                self.run_similarity_analysis(
                                    clusters, descriptions, table_names,
                                    method, param, feature_method, results
                                )
                        
                        except Exception as e:
                            self.logger.error(f"Error in clustering {method}_{param}: {e}")
                            continue
            
            except Exception as e:
                self.logger.error(f"Error processing feature method {feature_method}: {e}")
                continue
        
        return results
    
    def run_similarity_analysis(self, initial_clusters, descriptions, table_names,
                               method, param, feature_method, results):
        """
        Run similarity analysis using cached similarity scores
        """
        for similarity_threshold in config.clustering.similarity_thresholds:
            self.logger.info(f"Running similarity analysis with threshold: {similarity_threshold}")
            
            try:
                # Calculate similarities (will use cache if available)
                similarity_scores, num_pairs_computed = self.similarity_calculator.calculate_cluster_similarities(
                    initial_clusters, descriptions, max_workers=1  # Single thread for standalone
                )
                
                # Merge clusters based on similarity
                merged_clusters = self.clustering_manager.merger.merge_clusters_by_similarity(
                    initial_clusters, similarity_scores, similarity_threshold, table_names
                )
                
                # Evaluate merged results
                experiment_info = {
                    'experiment_type': 'clustering + API',
                    'clustering_method': method,
                    'clustering_param': param,
                    'feature_extraction_method': feature_method,
                    'similarity_threshold': similarity_threshold,
                    'num_pairs_to_compute': num_pairs_computed
                }
                
                metrics = self.evaluator.evaluate_clustering_result(
                    merged_clusters, table_names, experiment_info
                )
                
                key = f"{method}_{param}_{feature_method}_{similarity_threshold}"
                results['clustering_with_api'][key] = {
                    'clusters': merged_clusters,
                    'metrics': metrics,
                    'experiment_info': experiment_info,
                    'similarity_scores': similarity_scores
                }
                
            except Exception as e:
                self.logger.error(f"Error in similarity analysis: {e}")
                # Continue without similarity analysis
                continue
    
    def generate_merged_schemas(self, table_data: dict):
        """
        Generate merged table schemas from best clustering results
        """
        self.logger.info("Generating merged table schemas from best results")
        
        try:
            # Get all experiment results
            all_results = self.evaluator.experiment_tracker.get_results_dataframe().to_dict('records')
            
            # Get best clustering results for merging
            best_results = self.table_merger.get_best_merge_candidates(all_results)
            
            for result in best_results[:3]:  # Process top 3 results
                # Reconstruct clusters from the experiment
                experiment_info = {
                    'clustering_method': result.get('clustering_method'),
                    'clustering_param': result.get('clustering_param'),
                    'feature_extraction_method': result.get('feature_extraction_method'),
                    'similarity_threshold': result.get('similarity_threshold'),
                    'experiment_type': result.get('experiment_type'),
                    'f1_score': result.get('f1_score'),
                    'adjusted_rand_index': result.get('adjusted_rand_index')
                }
                
                # We need to reconstruct the actual clusters for this result
                # For now, use the grouping files that were already generated
                clusters = self._reconstruct_clusters_from_grouping(result, table_data.keys())
                
                if clusters:
                    # Merge table schemas
                    merged_schemas = self.table_merger.merge_table_schemas(
                        clusters, table_data, experiment_info
                    )
                    
                    # Save merged schemas
                    self.table_merger.save_merged_schemas(merged_schemas, experiment_info)
                    
                    # Create summary report
                    self.table_merger.create_merge_summary_report(merged_schemas, experiment_info)
            
            self.logger.info("Schema merging completed successfully")
        except Exception as e:
            self.logger.error(f"Error generating merged schemas: {e}")
    
    def _reconstruct_clusters_from_grouping(self, result: dict, all_tables: list) -> List[List[str]]:
        """
        Reconstruct clusters from grouping files
        """
        try:
            method = result.get('clustering_method')
            param = result.get('clustering_param')
            sim_threshold = result.get('similarity_threshold')
            
            # Generate the expected grouping filename
            if sim_threshold is not None:
                filename = f"groupings_{method}_sim_{sim_threshold}_param_{param}.csv"
            else:
                filename = f"groupings_{method}_param_{param}.csv"
            
            filepath = os.path.join('./analysis_outputs', filename)
            
            if os.path.exists(filepath):
                clusters = []
                import csv
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) >= 2 and row[1].strip():
                            tables_str = row[1].strip()
                            cluster_tables = [t.strip() for t in tables_str.split(',') if t.strip()]
                            if cluster_tables:
                                clusters.append(cluster_tables)
                
                self.logger.info(f"Reconstructed {len(clusters)} clusters from {filename}")
                return clusters
            else:
                self.logger.warning(f"Grouping file not found: {filepath}")
                return []
        except Exception as e:
            self.logger.error(f"Error reconstructing clusters: {e}")
            return []

    def generate_reports(self):
        """
        Generate analysis reports
        """
        self.logger.info("Generating analysis reports")
        
        try:
            # Finalize evaluation
            self.evaluator.finalize_evaluation()
            
            # Log statistics
            cache_stats = self.similarity_calculator.get_cache_stats()
            self.logger.info(f"Similarity cache statistics: {cache_stats}")
            
            self.logger.info("Reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def run_full_analysis(self) -> dict:
        """
        Run the complete analysis pipeline in standalone mode
        """
        self.logger.info("Starting full standalone analysis")
        
        try:
            # Step 1: Verify data availability
            if not self.verify_data_availability():
                raise RuntimeError("Required data files not available")
            
            # Step 2: Adjust file paths
            self.adjust_file_paths()
            
            # Step 3: Run clustering analysis
            results = self.run_clustering_analysis()
            
            # Step 4: Generate merged schemas from best results
            # Load table data for merging
            original_output_dir = config.paths.output_dir
            config.paths.output_dir = self.data_directory
            try:
                table_data = self.table_manager.load_all_table_data()
                self.generate_merged_schemas(table_data)
            finally:
                config.paths.output_dir = original_output_dir
            
            # Step 5: Generate reports
            self.generate_reports()
            
            # Summary
            clustering_only_count = len(results['clustering_only'])
            clustering_api_count = len(results['clustering_with_api'])
            
            self.logger.info("Standalone analysis completed successfully!")
            self.logger.info(f"Total experiments: {clustering_only_count + clustering_api_count}")
            self.logger.info(f"- Clustering only: {clustering_only_count}")
            self.logger.info(f"- Clustering + API: {clustering_api_count}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Standalone analysis failed: {e}")
            raise

def main():
    """Main entry point for standalone analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone MAUDE Schema Analysis')
    parser.add_argument('--data-dir', 
                       default='./data',
                       help='Directory containing table data JSON files')
    parser.add_argument('--clustering-only', action='store_true',
                       help='Run only clustering analysis (no API similarity)')
    parser.add_argument('--no-api', action='store_true',
                       help='Disable API calls (use cached similarities only)')
    
    args = parser.parse_args()
    
    # Disable API if requested
    if args.clustering_only or args.no_api:
        config.processing.enable_similarity_calculation = False
        print("API similarity calculation disabled")
    
    try:
        # Initialize analyzer
        analyzer = StandaloneMAUDEAnalyzer(args.data_dir)
        
        # Run analysis
        results = analyzer.run_full_analysis()
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {config.paths.evaluation_results_file}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()