"""
Main application for MAUDE Schema Compressor
Coordinates the entire schema analysis and compression pipeline
"""
import logging
import argparse
import sys
from typing import Dict, List, Any
from tqdm import tqdm

# Import all modules
from config import config
from database import DatabaseManager, TableDataManager
from features import FeatureProcessor
from clustering import ClusteringManager
from similarity import SimilarityCalculator
from evaluation import Evaluator

class MAUDESchemaCompressor:
    """Main application class for MAUDE Schema Compressor"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.table_manager = TableDataManager(self.db_manager)
        self.feature_processor = FeatureProcessor()
        self.clustering_manager = ClusteringManager()
        self.similarity_calculator = SimilarityCalculator()
        self.evaluator = Evaluator()
        
        self.logger.info("MAUDE Schema Compressor initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('maude_compressor.log')
            ]
        )
    
    def extract_database_schema(self, force_refresh: bool = False) -> bool:
        """
        Extract database schema and sample data
        """
        self.logger.info("Starting database schema extraction")
        
        # Check if data already exists
        if not force_refresh:
            try:
                existing_data = self.table_manager.load_all_table_data()
                if existing_data:
                    self.logger.info(f"Found existing data for {len(existing_data)} tables")
                    return True
            except Exception as e:
                self.logger.warning(f"Error loading existing data: {e}")
        
        # Connect to database
        if not self.db_manager.connect():
            self.logger.error("Failed to connect to database")
            return False
        
        try:
            # Extract and save all table information
            success = self.table_manager.extract_and_save_all_tables()
            if success:
                self.logger.info("Database schema extraction completed successfully")
            return success
        finally:
            self.db_manager.close()
    
    def run_clustering_experiments(self) -> Dict[str, Any]:
        """
        Run clustering experiments with different methods and parameters
        """
        self.logger.info("Starting clustering experiments")
        
        # Load table data and generate descriptions
        table_data = self.table_manager.load_all_table_data()
        if not table_data:
            raise RuntimeError("No table data available. Run extract_database_schema first.")
        
        descriptions = self.table_manager.get_all_table_descriptions()
        
        # Prepare table structures for feature extraction
        table_structures = {}
        for table_name, data in table_data.items():
            table_structures[table_name] = data.get('structure', [])
        
        results = {'clustering_only': {}, 'clustering_with_api': {}}
        
        # Run experiments for each feature extraction method
        for feature_method in config.processing.feature_extraction_methods:
            self.logger.info(f"Processing with feature extraction method: {feature_method}")
            
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
                            
                            # If API similarity calculation is enabled, run clustering + API experiments
                            if config.processing.enable_similarity_calculation:
                                self.run_clustering_with_api(
                                    clusters, descriptions, table_names,
                                    method, param, feature_method, results
                                )
                        
                        except Exception as e:
                            self.logger.error(f"Error in clustering experiment {method}_{param}: {e}")
                            continue
            
            except Exception as e:
                self.logger.error(f"Error processing feature method {feature_method}: {e}")
                continue
        
        return results
    
    def run_clustering_with_api(self, initial_clusters: List[List[str]], 
                               descriptions: Dict[str, str],
                               table_names: List[str],
                               method: str, param: Any, feature_method: str,
                               results: Dict[str, Any]):
        """
        Run clustering + API similarity experiments
        """
        for similarity_threshold in config.clustering.similarity_thresholds:
            self.logger.info(f"Running clustering + API with similarity threshold: {similarity_threshold}")
            
            try:
                # Calculate similarities within clusters
                similarity_scores, num_pairs_computed = self.similarity_calculator.calculate_cluster_similarities(
                    initial_clusters, descriptions, max_workers=3
                )
                
                # Merge clusters based on similarity
                merged_clusters = self.clustering_manager.merger.merge_clusters_by_similarity(
                    initial_clusters, similarity_scores, similarity_threshold, table_names
                )
                
                # Evaluate merged clustering results
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
                self.logger.error(f"Error in API similarity experiment: {e}")
    
    def generate_reports(self):
        """
        Generate analysis reports and visualizations
        """
        self.logger.info("Generating analysis reports")
        
        try:
            # Finalize evaluation (saves results and generates summary)
            self.evaluator.finalize_evaluation()
            
            # Log cache statistics
            cache_stats = self.similarity_calculator.get_cache_stats()
            self.logger.info(f"Similarity cache statistics: {cache_stats}")
            
            self.logger.info("Reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def run_full_pipeline(self, force_refresh: bool = False):
        """
        Run the complete MAUDE schema compression pipeline
        """
        self.logger.info("Starting full MAUDE Schema Compression pipeline")
        
        try:
            # Step 1: Extract database schema
            if not self.extract_database_schema(force_refresh):
                raise RuntimeError("Failed to extract database schema")
            
            # Step 2: Run clustering experiments
            results = self.run_clustering_experiments()
            
            # Step 3: Generate reports
            self.generate_reports()
            
            # Summary
            clustering_only_count = len(results['clustering_only'])
            clustering_api_count = len(results['clustering_with_api'])
            
            self.logger.info(f"Pipeline completed successfully!")
            self.logger.info(f"Total experiments: {clustering_only_count + clustering_api_count}")
            self.logger.info(f"- Clustering only: {clustering_only_count}")
            self.logger.info(f"- Clustering + API: {clustering_api_count}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MAUDE Schema Compressor')
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract database schema, do not run experiments')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of extracted data')
    parser.add_argument('--clustering-only', action='store_true',
                       help='Run only clustering experiments (no API similarity)')
    parser.add_argument('--config-info', action='store_true',
                       help='Display configuration information')
    
    args = parser.parse_args()
    
    # Display configuration info if requested
    if args.config_info:
        print("MAUDE Schema Compressor Configuration:")
        print(f"Database: {config.database.host}:{config.database.port}/{config.database.database}")
        print(f"Output directory: {config.paths.output_dir}")
        print(f"Feature extraction methods: {config.processing.feature_extraction_methods}")
        print(f"Clustering methods: {list(config.clustering.clustering_methods.keys())}")
        print(f"Similarity thresholds: {config.clustering.similarity_thresholds}")
        print(f"API enabled: {bool(config.api.deepseek_api_key)}")
        return
    
    try:
        # Initialize compressor
        compressor = MAUDESchemaCompressor()
        
        if args.extract_only:
            # Only extract database schema
            compressor.extract_database_schema(args.force_refresh)
        elif args.clustering_only:
            # Temporarily disable API similarity calculation
            original_enable_similarity = config.processing.enable_similarity_calculation
            config.processing.enable_similarity_calculation = False
            try:
                compressor.run_full_pipeline(args.force_refresh)
            finally:
                config.processing.enable_similarity_calculation = original_enable_similarity
        else:
            # Run full pipeline
            compressor.run_full_pipeline(args.force_refresh)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()