"""
Configuration module for MAUDE Schema Compressor
Contains all configuration settings and parameters
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    database: str = os.getenv('DB_NAME', 'maude')
    user: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '12345687')
    host: str = os.getenv('DB_HOST', '192.168.8.167')
    port: str = os.getenv('DB_PORT', '5432')
    schema: str = 'maude'

@dataclass
class APIConfig:
    """API configuration for similarity calculations"""
    deepseek_api_key: str = os.getenv('DEEPSEEK_API_KEY', '')
    deepseek_base_url: str = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    retry_limit: int = int(os.getenv('RETRY_LIMIT', '2'))
    max_tokens: int = int(os.getenv('MAX_TOKENS', '100000'))
    timeout: int = 300

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    sample_size: int = 3
    random_seed: int = int(os.getenv('RANDOM_SEED', '42'))
    overwrite_existing: bool = os.getenv('OVERWRITE_EXISTING', 'True').lower() in ['true', '1', 't']
    enable_pca: bool = os.getenv('ENABLE_PCA', 'True').lower() in ['true', '1', 't']
    enable_similarity_calculation: bool = os.getenv('ENABLE_SIMILARITY_CALCULATION', 'True').lower() in ['true', '1', 't']
    
    # Feature extraction methods
    feature_extraction_methods: List[str] = None
    
    def __post_init__(self):
        if self.feature_extraction_methods is None:
            self.feature_extraction_methods = ['tfidf', 'sentence_transformer']

@dataclass
class ClusteringConfig:
    """Clustering configuration"""
    similarity_thresholds: List[float] = None
    prefilter_jaccard_threshold: float = 0.1
    distance_thresholds: List[float] = None
    use_hierarchical_clustering: bool = True
    
    # Clustering methods and parameters
    clustering_methods: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.similarity_thresholds is None:
            self.similarity_thresholds = [0.7, 0.8, 0.9]
        
        if self.distance_thresholds is None:
            self.distance_thresholds = [1.0]
        
        if self.clustering_methods is None:
            self.clustering_methods = {
                'kmeans_manual': {
                    'params': [3, 4, 5, 6, 7],
                    'description': 'K-Means clustering with manually specified cluster counts'
                },
                'kmeans_auto': {
                    'params': [None],
                    'description': 'K-Means clustering with automatic detection of optimal cluster count'
                },
                'hierarchical': {
                    'params': [0.8, 1.0, 1.2],
                    'description': 'Hierarchical clustering with varying distance thresholds'
                },
                'dbscan_manual': {
                    'params': [(0.5, 5), (0.6, 5), (0.7, 5)],
                    'description': 'DBSCAN clustering with varying ε values and min_samples=5'
                },
                'dbscan_auto': {
                    'params': [None],
                    'description': 'DBSCAN clustering with automatic parameter selection'
                }
            }

@dataclass
class PathConfig:
    """File path configuration"""
    output_dir: str = os.getenv('OUTPUT_DIR', 'maude_schema_analysis')
    post_merged_dir: str = os.getenv('POST_MERGED_DIR', './merged_schemas')
    combined_output_file: str = os.getenv('COMBINED_OUTPUT_FILE', 'maude_schema_combined.txt')
    cache_file: str = os.getenv('CACHE_FILE', 'similarity_cache.json')
    context_file: str = os.getenv('GLOBAL_CONTEXT_FILE', 'context.txt')
    analysis_file: str = os.getenv('OUTPUT_ANALYSIS_FILE', 'maude_schema_analysis.json')
    manual_grouping_file: str = os.getenv('MANUAL_GROUPING_FILE', 'manual_grouping.json')
    evaluation_results_file: str = os.getenv('EVALUATION_RESULTS_FILE', 'evaluation_results.csv')

class Config:
    """Main configuration class that combines all configurations"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.processing = ProcessingConfig()
        self.clustering = ClusteringConfig()
        self.paths = PathConfig()
        
        # Validate configuration
        self._validate_config()
        
        # Create necessary directories
        self._create_directories()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if not self.api.deepseek_api_key and self.processing.enable_similarity_calculation:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("DEEPSEEK_API_KEY not provided. Will use cached similarities only.")
            # Keep similarity calculation enabled to use cached data
        
        if self.processing.sample_size < 1:
            raise ValueError("Sample size must be at least 1")
        
        if not self.clustering.similarity_thresholds:
            raise ValueError("At least one similarity threshold must be specified")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.paths.output_dir,
            self.paths.post_merged_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_database_uri(self) -> str:
        """Get database connection URI"""
        return (f"postgresql://{self.database.user}:{self.database.password}@"
                f"{self.database.host}:{self.database.port}/{self.database.database}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'processing': self.processing.__dict__,
            'clustering': self.clustering.__dict__,
            'paths': self.paths.__dict__
        }

# Global configuration instance
config = Config()