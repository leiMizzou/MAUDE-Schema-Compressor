"""
Feature extraction module for MAUDE Schema Compressor
Handles different feature extraction methods including TF-IDF and Sentence Transformers
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import config

class FeatureExtractor:
    """Base class for feature extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, descriptions: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from table descriptions
        Returns feature matrix and list of table names
        """
        raise NotImplementedError("Subclasses must implement extract_features method")

class TFIDFExtractor(FeatureExtractor):
    """TF-IDF feature extractor"""
    
    def __init__(self, max_features: int = 5000, stop_words: str = 'english'):
        super().__init__()
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = None
    
    def extract_features(self, descriptions: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract TF-IDF features from table descriptions
        """
        table_names = list(descriptions.keys())
        corpus = [descriptions[table] for table in table_names]
        
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=self.max_features,
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        
        try:
            feature_matrix = self.vectorizer.fit_transform(corpus).toarray()
            self.logger.info(f"TF-IDF feature extraction completed. Shape: {feature_matrix.shape}")
            return feature_matrix, table_names
        except Exception as e:
            self.logger.error(f"Error in TF-IDF feature extraction: {e}")
            raise

class SentenceTransformerExtractor(FeatureExtractor):
    """Sentence Transformer feature extractor"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package is required for SentenceTransformerExtractor")
        
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Error loading SentenceTransformer model: {e}")
                raise
    
    def extract_features(self, descriptions: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract sentence embeddings from table descriptions
        """
        self._load_model()
        
        table_names = list(descriptions.keys())
        corpus = [descriptions[table] for table in table_names]
        
        try:
            embeddings = self.model.encode(corpus, show_progress_bar=True)
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            self.logger.info(f"SentenceTransformer feature extraction completed. Shape: {embeddings_scaled.shape}")
            return embeddings_scaled, table_names
        except Exception as e:
            self.logger.error(f"Error in SentenceTransformer feature extraction: {e}")
            raise

class StructuredFeatureExtractor:
    """Extractor for structured features from table metadata"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_structured_features(self, table_structures: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Extract structured features from table structures
        Returns DataFrame with table features
        """
        features = []
        
        for table_name, structure in table_structures.items():
            # Basic statistics
            num_columns = len(structure)
            
            # Count different column types
            type_counts = {}
            nullable_count = 0
            has_default_count = 0
            
            for column in structure:
                # Column type categorization
                col_type = str(column.get('type', '')).lower()
                if 'int' in col_type or 'serial' in col_type:
                    type_category = 'integer'
                elif 'char' in col_type or 'text' in col_type or 'varchar' in col_type:
                    type_category = 'text'
                elif 'date' in col_type or 'time' in col_type:
                    type_category = 'datetime'
                elif 'bool' in col_type:
                    type_category = 'boolean'
                elif 'float' in col_type or 'real' in col_type or 'double' in col_type:
                    type_category = 'float'
                else:
                    type_category = 'other'
                
                type_counts[type_category] = type_counts.get(type_category, 0) + 1
                
                # Count nullable and default columns
                if column.get('nullable', True):
                    nullable_count += 1
                if column.get('default') is not None:
                    has_default_count += 1
            
            # Calculate ratios
            nullable_ratio = nullable_count / num_columns if num_columns > 0 else 0
            has_default_ratio = has_default_count / num_columns if num_columns > 0 else 0
            
            feature_row = {
                'table_name': table_name,
                'num_columns': num_columns,
                'nullable_ratio': nullable_ratio,
                'has_default_ratio': has_default_ratio,
                'integer_columns': type_counts.get('integer', 0),
                'text_columns': type_counts.get('text', 0),
                'datetime_columns': type_counts.get('datetime', 0),
                'boolean_columns': type_counts.get('boolean', 0),
                'float_columns': type_counts.get('float', 0),
                'other_columns': type_counts.get('other', 0)
            }
            
            features.append(feature_row)
        
        df_features = pd.DataFrame(features).set_index('table_name')
        self.logger.info(f"Extracted structured features for {len(df_features)} tables")
        return df_features

class FeatureProcessor:
    """Main feature processing class that combines different extraction methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.structured_extractor = StructuredFeatureExtractor()
    
    def get_feature_extractor(self, method: str) -> FeatureExtractor:
        """
        Get feature extractor based on method name
        """
        if method == 'tfidf':
            return TFIDFExtractor()
        elif method == 'sentence_transformer':
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.warning("sentence-transformers not available, falling back to TF-IDF")
                return TFIDFExtractor()
            return SentenceTransformerExtractor()
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
    
    def extract_combined_features(self, descriptions: Dict[str, str], 
                                table_structures: Dict[str, List[Dict]],
                                method: str = 'tfidf') -> Tuple[np.ndarray, List[str]]:
        """
        Extract and combine textual and structured features
        """
        # Extract textual features
        extractor = self.get_feature_extractor(method)
        text_features, table_names = extractor.extract_features(descriptions)
        
        # Extract structured features
        structured_features = self.structured_extractor.extract_structured_features(table_structures)
        
        # Ensure table order consistency
        try:
            structured_matrix = structured_features.loc[table_names].values
        except KeyError as e:
            self.logger.error(f"Table name mismatch between textual and structured features: {e}")
            # Use intersection of table names
            common_tables = list(set(table_names) & set(structured_features.index))
            table_names = common_tables
            text_features = text_features[:len(common_tables)]
            structured_matrix = structured_features.loc[common_tables].values
        
        # Combine features
        combined_features = np.hstack((text_features, structured_matrix))
        
        self.logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features, table_names
    
    def apply_dimensionality_reduction(self, features: np.ndarray, 
                                     n_components: int = 100) -> np.ndarray:
        """
        Apply PCA dimensionality reduction
        """
        if not config.processing.enable_pca:
            self.logger.info("PCA disabled in configuration")
            return features
        
        if features.shape[1] <= n_components:
            self.logger.info(f"Feature dimension ({features.shape[1]}) <= n_components ({n_components}), skipping PCA")
            return features
        
        try:
            # Standardize features to avoid numerical issues
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Replace NaN/Inf values with 0
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            pca = PCA(n_components=n_components, random_state=config.processing.random_seed)
            reduced_features = pca.fit_transform(features_scaled)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            self.logger.info(f"PCA applied: {features.shape} -> {reduced_features.shape}, "
                           f"explained variance: {explained_variance:.3f}")
            
            return reduced_features
        except Exception as e:
            self.logger.error(f"Error applying PCA: {e}")
            return features
    
    def process_features(self, descriptions: Dict[str, str],
                        table_structures: Dict[str, List[Dict]],
                        method: str = 'tfidf') -> Tuple[np.ndarray, List[str]]:
        """
        Main feature processing pipeline
        """
        self.logger.info(f"Starting feature processing with method: {method}")
        
        # Extract combined features
        features, table_names = self.extract_combined_features(
            descriptions, table_structures, method
        )
        
        # Apply dimensionality reduction if enabled
        features = self.apply_dimensionality_reduction(features)
        
        self.logger.info(f"Feature processing completed. Final shape: {features.shape}")
        return features, table_names