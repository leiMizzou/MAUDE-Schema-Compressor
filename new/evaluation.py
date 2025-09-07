"""
Evaluation module for MAUDE Schema Compressor
Handles evaluation metrics and comparison with ground truth
"""
import json
import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

from config import config

class EvaluationMetrics:
    """Calculates various clustering evaluation metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_pairwise_metrics(self, manual_labels: List[int], 
                                 predicted_labels: List[int]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score based on pairwise comparisons
        """
        n = len(manual_labels)
        if n != len(predicted_labels):
            raise ValueError("Manual and predicted labels must have the same length")
        
        # Generate all pairs and their ground truth/predicted relationships
        manual_pairs = set()
        predicted_pairs = set()
        
        for i, j in combinations(range(n), 2):
            if manual_labels[i] == manual_labels[j]:
                manual_pairs.add((i, j))
            if predicted_labels[i] == predicted_labels[j]:
                predicted_pairs.add((i, j))
        
        # Calculate metrics
        tp = len(manual_pairs & predicted_pairs)  # True positives
        fp = len(predicted_pairs - manual_pairs)  # False positives
        fn = len(manual_pairs - predicted_pairs)  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def calculate_clustering_metrics(self, manual_labels: List[int], 
                                   predicted_labels: List[int]) -> Dict[str, float]:
        """
        Calculate standard clustering evaluation metrics
        """
        metrics = {}
        
        # Adjusted Rand Index
        try:
            ari = adjusted_rand_score(manual_labels, predicted_labels)
            metrics['adjusted_rand_index'] = ari
        except Exception as e:
            self.logger.warning(f"Error calculating ARI: {e}")
            metrics['adjusted_rand_index'] = 0.0
        
        # Normalized Mutual Information
        try:
            nmi = normalized_mutual_info_score(manual_labels, predicted_labels)
            metrics['normalized_mutual_info'] = nmi
        except Exception as e:
            self.logger.warning(f"Error calculating NMI: {e}")
            metrics['normalized_mutual_info'] = 0.0
        
        # Pairwise metrics
        try:
            pairwise_metrics = self.calculate_pairwise_metrics(manual_labels, predicted_labels)
            metrics.update(pairwise_metrics)
        except Exception as e:
            self.logger.warning(f"Error calculating pairwise metrics: {e}")
            metrics.update({
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            })
        
        return metrics
    
    def clusters_to_labels(self, clusters: List[List[str]], 
                          table_names: List[str]) -> List[int]:
        """
        Convert cluster assignments to label array
        """
        labels = [-1] * len(table_names)  # Initialize with -1 (no cluster)
        
        for cluster_id, cluster in enumerate(clusters):
            for table in cluster:
                if table in table_names:
                    table_idx = table_names.index(table)
                    labels[table_idx] = cluster_id
        
        return labels
    
    def labels_to_clusters(self, labels: List[int], 
                          table_names: List[str]) -> List[List[str]]:
        """
        Convert label array to cluster assignments
        """
        clusters = {}
        for table, label in zip(table_names, labels):
            if label >= 0:  # Ignore unassigned tables
                clusters.setdefault(label, []).append(table)
        
        return list(clusters.values())

class GroundTruthManager:
    """Manages ground truth data for evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Don't store the path at initialization, get it dynamically
    
    def load_manual_grouping(self) -> Dict[str, int]:
        """
        Load manual grouping from file
        Returns dictionary mapping table names to group IDs
        """
        # Get current path from config (may have been updated)
        manual_grouping_file = config.paths.manual_grouping_file
        
        try:
            with open(manual_grouping_file, 'r', encoding='utf-8') as f:
                manual_grouping = json.load(f)
            self.logger.info(f"Loaded manual grouping for {len(manual_grouping)} tables from {manual_grouping_file}")
            return manual_grouping
        except FileNotFoundError:
            self.logger.warning(f"Manual grouping file '{manual_grouping_file}' not found")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding manual grouping file: {e}")
            return {}
    
    def save_manual_grouping(self, grouping: Dict[str, int]):
        """
        Save manual grouping to file
        """
        manual_grouping_file = config.paths.manual_grouping_file
        try:
            with open(manual_grouping_file, 'w', encoding='utf-8') as f:
                json.dump(grouping, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved manual grouping for {len(grouping)} tables")
        except Exception as e:
            self.logger.error(f"Error saving manual grouping: {e}")
    
    def create_initial_grouping(self, ai_grouping: Dict[str, int]) -> Dict[str, int]:
        """
        Create initial manual grouping file based on AI results
        """
        existing_grouping = self.load_manual_grouping()
        if existing_grouping:
            self.logger.info("Manual grouping already exists")
            return existing_grouping
        
        # Create initial grouping based on AI results
        self.save_manual_grouping(ai_grouping)
        self.logger.info("Created initial manual grouping based on AI results")
        return ai_grouping
    
    def validate_grouping(self, grouping: Dict[str, int], 
                         table_names: List[str]) -> bool:
        """
        Validate that grouping contains all required tables
        """
        missing_tables = set(table_names) - set(grouping.keys())
        extra_tables = set(grouping.keys()) - set(table_names)
        
        if missing_tables:
            self.logger.warning(f"Missing tables in grouping: {missing_tables}")
        if extra_tables:
            self.logger.warning(f"Extra tables in grouping: {extra_tables}")
        
        return len(missing_tables) == 0

class ExperimentTracker:
    """Tracks and stores experiment results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = []
    
    def add_experiment(self, experiment_type: str, clustering_method: str,
                      clustering_param: Any, feature_extraction_method: str,
                      similarity_threshold: Optional[float], num_clusters: int,
                      num_pairs_to_compute: int, metrics: Dict[str, float]):
        """
        Add experiment result to tracker
        """
        result = {
            'experiment_type': experiment_type,
            'clustering_method': clustering_method,
            'clustering_param': clustering_param,
            'feature_extraction_method': feature_extraction_method,
            'similarity_threshold': similarity_threshold,
            'num_clusters': num_clusters,
            'num_pairs_to_compute': num_pairs_to_compute,
            **metrics  # Unpack all metrics
        }
        
        self.results.append(result)
        self.logger.info(f"Added experiment result: {experiment_type} - {clustering_method}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get all results as pandas DataFrame
        """
        return pd.DataFrame(self.results)
    
    def save_results(self, filename: str = None):
        """
        Save results to CSV file
        """
        if filename is None:
            filename = config.paths.evaluation_results_file
        
        try:
            df = self.get_results_dataframe()
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"Saved {len(self.results)} experiment results to '{filename}'")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def load_results(self, filename: str = None) -> bool:
        """
        Load existing results from CSV file
        Returns True if successful, False otherwise
        """
        if filename is None:
            filename = config.paths.evaluation_results_file
        
        try:
            df = pd.read_csv(filename)
            self.results = df.to_dict('records')
            self.logger.info(f"Loaded {len(self.results)} experiment results from '{filename}'")
            return True
        except FileNotFoundError:
            self.logger.info("No existing results file found")
            return False
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False
    
    def clear_results(self):
        """Clear all stored results"""
        self.results = []
        self.logger.info("Cleared all experiment results")

class GroupingExporter:
    """Exports clustering results in various formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def save_grouping_to_csv(self, grouping: Dict[str, int], 
                           experiment_info: Dict[str, Any],
                           output_dir: str = None) -> str:
        """
        Save grouping results to CSV file
        Returns the filename of the saved file
        """
        if output_dir is None:
            output_dir = config.paths.output_dir
        
        # Create group-to-tables mapping
        group_to_tables = {}
        for table, group_id in grouping.items():
            group_to_tables.setdefault(group_id, []).append(table)
        
        # Create DataFrame
        data = []
        for group_id, tables in group_to_tables.items():
            data.append({
                'Group ID': group_id,
                'Tables': ', '.join(sorted(tables)),
                'Table Count': len(tables)
            })
        
        df = pd.DataFrame(data).sort_values('Group ID')
        
        # Generate filename
        method = experiment_info.get('clustering_method', 'unknown')
        param = experiment_info.get('clustering_param', 'none')
        sim_threshold = experiment_info.get('similarity_threshold')
        
        if sim_threshold is not None:
            filename = f"groupings_{method}_sim_{sim_threshold}_param_{param}.csv"
        else:
            filename = f"groupings_{method}_param_{param}.csv"
        
        filepath = f"{output_dir}/{filename}"
        
        try:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            self.logger.info(f"Saved grouping results to '{filepath}'")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving grouping to CSV: {e}")
            return ""
    
    def export_summary_statistics(self, all_results: List[Dict[str, Any]], 
                                output_file: str = None) -> str:
        """
        Export summary statistics for all experiments
        """
        if output_file is None:
            output_file = f"{config.paths.output_dir}/experiment_summary.json"
        
        # Calculate summary statistics
        summary = {
            'total_experiments': len(all_results),
            'clustering_methods': list(set(r['clustering_method'] for r in all_results)),
            'feature_extraction_methods': list(set(r['feature_extraction_method'] for r in all_results)),
            'experiment_types': list(set(r['experiment_type'] for r in all_results)),
        }
        
        # Add performance statistics
        if all_results:
            metrics = ['adjusted_rand_index', 'normalized_mutual_info', 'f1_score']
            for metric in metrics:
                values = [r.get(metric, 0) for r in all_results if metric in r]
                if values:
                    summary[f'{metric}_stats'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved experiment summary to '{output_file}'")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving summary statistics: {e}")
            return ""

class Evaluator:
    """Main evaluation class that coordinates all evaluation activities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = EvaluationMetrics()
        self.ground_truth_manager = GroundTruthManager()
        self.experiment_tracker = ExperimentTracker()
        self.grouping_exporter = GroupingExporter()
    
    def evaluate_clustering_result(self, clusters: List[List[str]], 
                                 table_names: List[str],
                                 experiment_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate clustering result against ground truth
        """
        # Convert clusters to grouping
        ai_grouping = {}
        for cluster_id, cluster in enumerate(clusters):
            for table in cluster:
                ai_grouping[table] = cluster_id
        
        # Load manual grouping and calculate metrics if available
        manual_grouping = self.ground_truth_manager.load_manual_grouping()
        metrics = {}
        
        if manual_grouping:
            # Ensure we have the same tables in both groupings
            common_tables = list(set(manual_grouping.keys()) & set(ai_grouping.keys()))
            if common_tables:
                # Create labels for common tables
                manual_labels = [manual_grouping[table] for table in common_tables]
                ai_labels = [ai_grouping[table] for table in common_tables]
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_clustering_metrics(manual_labels, ai_labels)
            else:
                self.logger.error("No common tables between manual and AI groupings")
        else:
            self.logger.warning("No manual grouping available for evaluation")
            # Add placeholder metrics
            metrics = {
                'adjusted_rand_index': -1.0,
                'normalized_mutual_info': -1.0,
                'precision': -1.0,
                'recall': -1.0,
                'f1_score': -1.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        
        # Always add experiment to tracker
        self.experiment_tracker.add_experiment(
            experiment_type=experiment_info.get('experiment_type', 'unknown'),
            clustering_method=experiment_info.get('clustering_method', 'unknown'),
            clustering_param=experiment_info.get('clustering_param'),
            feature_extraction_method=experiment_info.get('feature_extraction_method', 'unknown'),
            similarity_threshold=experiment_info.get('similarity_threshold'),
            num_clusters=len(clusters),
            num_pairs_to_compute=experiment_info.get('num_pairs_to_compute', 0),
            metrics=metrics
        )
        
        # Export grouping results
        self.grouping_exporter.save_grouping_to_csv(ai_grouping, experiment_info)
        
        return metrics
    
    def finalize_evaluation(self):
        """
        Finalize evaluation by saving all results and generating summary
        """
        # Save experiment results
        self.experiment_tracker.save_results()
        
        # Generate summary statistics
        all_results = self.experiment_tracker.get_results_dataframe().to_dict('records')
        self.grouping_exporter.export_summary_statistics(all_results)
        
        self.logger.info("Evaluation finalized successfully")