"""
Clustering module for MAUDE Schema Compressor
Implements various clustering algorithms for table grouping
"""
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from config import config

class ClusteringAlgorithm:
    """Base class for clustering algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fit_predict(self, features: np.ndarray, table_names: List[str], 
                   param: Optional[any] = None) -> List[List[str]]:
        """
        Fit clustering algorithm and return clusters
        Returns list of clusters, where each cluster is a list of table names
        """
        raise NotImplementedError("Subclasses must implement fit_predict method")

class KMeansClusterer(ClusteringAlgorithm):
    """K-Means clustering implementation"""
    
    def fit_predict(self, features: np.ndarray, table_names: List[str], 
                   param: Optional[int] = None) -> List[List[str]]:
        """
        Perform K-Means clustering
        param: number of clusters (None for automatic detection)
        """
        if param is None:
            # Automatic cluster number detection
            n_clusters = self._find_optimal_clusters(features)
            if n_clusters is None:
                self.logger.warning("Could not determine optimal clusters, using default k=5")
                n_clusters = 5
        else:
            n_clusters = param
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=config.processing.random_seed)
            labels = kmeans.fit_predict(features)
            
            clusters = self._labels_to_clusters(labels, table_names)
            self.logger.info(f"K-Means clustering completed with {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in K-Means clustering: {e}")
            return [[table] for table in table_names]  # Fallback: each table in its own cluster
    
    def _find_optimal_clusters(self, features: np.ndarray, min_k: int = 2, max_k: int = 20) -> Optional[int]:
        """
        Find optimal number of clusters using silhouette score
        """
        best_k = None
        best_score = -1
        
        max_k = min(max_k, len(features) - 1)  # Ensure max_k doesn't exceed data size
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=config.processing.random_seed)
                labels = kmeans.fit_predict(features)
                
                if len(set(labels)) == 1:
                    continue  # Skip if all points in one cluster
                
                score = silhouette_score(features, labels)
                self.logger.debug(f"K={k}, Silhouette Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                self.logger.warning(f"Error evaluating K={k}: {e}")
                continue
        
        if best_k is not None:
            self.logger.info(f"Optimal K-Means clusters: {best_k}, Score: {best_score:.4f}")
        
        return best_k
    
    def _labels_to_clusters(self, labels: np.ndarray, table_names: List[str]) -> List[List[str]]:
        """Convert cluster labels to list of clusters"""
        clusters = {}
        for table, label in zip(table_names, labels):
            clusters.setdefault(label, []).append(table)
        return list(clusters.values())

class HierarchicalClusterer(ClusteringAlgorithm):
    """Hierarchical clustering implementation"""
    
    def fit_predict(self, features: np.ndarray, table_names: List[str], 
                   param: Optional[float] = None) -> List[List[str]]:
        """
        Perform hierarchical clustering
        param: distance threshold
        """
        if param is None:
            param = 1.0  # Default distance threshold
        
        try:
            # Compute distance matrix
            distance_matrix = cosine_distances(features)
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage='average',
                distance_threshold=param
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            clusters = self._labels_to_clusters(labels, table_names)
            self.logger.info(f"Hierarchical clustering completed with {len(clusters)} clusters "
                           f"(threshold: {param})")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical clustering: {e}")
            return [[table] for table in table_names]
    
    def _labels_to_clusters(self, labels: np.ndarray, table_names: List[str]) -> List[List[str]]:
        """Convert cluster labels to list of clusters"""
        clusters = {}
        for table, label in zip(table_names, labels):
            clusters.setdefault(label, []).append(table)
        return list(clusters.values())

class DBSCANClusterer(ClusteringAlgorithm):
    """DBSCAN clustering implementation"""
    
    def fit_predict(self, features: np.ndarray, table_names: List[str], 
                   param: Optional[Tuple[float, int]] = None) -> List[List[str]]:
        """
        Perform DBSCAN clustering
        param: tuple of (eps, min_samples) or None for automatic parameter selection
        """
        if param is None:
            eps = self._find_optimal_eps(features)
            min_samples = 5
        else:
            eps, min_samples = param
        
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = clustering.fit_predict(features)
            
            clusters = self._labels_to_clusters(labels, table_names)
            noise_points = len([label for label in labels if label == -1])
            
            self.logger.info(f"DBSCAN clustering completed with {len(clusters)} clusters "
                           f"(eps: {eps}, min_samples: {min_samples}, noise points: {noise_points})")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            return [[table] for table in table_names]
    
    def _find_optimal_eps(self, features: np.ndarray, k: int = 4) -> float:
        """
        Find optimal eps value using k-distance graph
        """
        try:
            neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
            neighbors_fit = neighbors.fit(features)
            distances, _ = neighbors_fit.kneighbors(features)
            distances = np.sort(distances[:, k-1], axis=0)
            
            # Find elbow point
            diff = np.diff(distances)
            diff2 = np.diff(diff)
            if len(diff2) > 0:
                elbow_index = np.argmax(diff2) + 2
                eps = distances[elbow_index] if elbow_index < len(distances) else distances[-1]
            else:
                eps = 0.5  # Default fallback
            
            self.logger.info(f"Optimal DBSCAN eps: {eps}")
            return eps
            
        except Exception as e:
            self.logger.error(f"Error finding optimal eps: {e}")
            return 0.5  # Default fallback
    
    def _labels_to_clusters(self, labels: np.ndarray, table_names: List[str]) -> List[List[str]]:
        """Convert cluster labels to list of clusters (ignoring noise points)"""
        clusters = {}
        for table, label in zip(table_names, labels):
            if label == -1:  # Ignore noise points
                continue
            clusters.setdefault(label, []).append(table)
        return list(clusters.values())

class ClusterMerger:
    """Handles merging clusters based on similarity scores"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge_clusters_by_similarity(self, clusters: List[List[str]], 
                                   similarity_scores: Dict[Tuple[str, str], float],
                                   threshold: float,
                                   all_tables: List[str]) -> List[List[str]]:
        """
        Merge clusters based on similarity threshold using graph connectivity
        """
        # Create graph with all tables as nodes
        G = nx.Graph()
        G.add_nodes_from(all_tables)
        
        # Add edges for table pairs that meet similarity threshold
        for (table1, table2), score in similarity_scores.items():
            if score >= threshold:
                G.add_edge(table1, table2)
        
        # Get connected components as merged clusters
        merged_clusters = [list(component) for component in nx.connected_components(G)]
        
        self.logger.info(f"Merged {len(clusters)} initial clusters into {len(merged_clusters)} "
                        f"clusters using threshold {threshold}")
        
        return merged_clusters
    
    def calculate_jaccard_similarity(self, table1_fields: set, table2_fields: set) -> float:
        """Calculate Jaccard similarity between two sets of fields"""
        intersection = table1_fields.intersection(table2_fields)
        union = table1_fields.union(table2_fields)
        return len(intersection) / len(union) if union else 0.0
    
    def prefilter_pairs(self, clusters: List[List[str]], 
                       descriptions: Dict[str, str],
                       threshold: float = None) -> List[Tuple[str, str]]:
        """
        Pre-filter table pairs using Jaccard similarity to reduce API calls
        """
        if threshold is None:
            threshold = config.clustering.prefilter_jaccard_threshold
        
        pairs_to_process = []
        total_pairs = 0
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            n = len(cluster)
            for i in range(n):
                for j in range(i + 1, n):
                    total_pairs += 1
                    table1, table2 = cluster[i], cluster[j]
                    
                    # Extract field names from descriptions
                    try:
                        desc1 = descriptions[table1]
                        desc2 = descriptions[table2]
                        
                        # Simple field extraction from description
                        fields1 = set(self._extract_fields_from_description(desc1))
                        fields2 = set(self._extract_fields_from_description(desc2))
                        
                        jaccard = self.calculate_jaccard_similarity(fields1, fields2)
                        
                        if jaccard >= threshold:
                            pair = tuple(sorted((table1, table2)))
                            pairs_to_process.append(pair)
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing pair ({table1}, {table2}): {e}")
                        # Include pair anyway if we can't filter
                        pair = tuple(sorted((table1, table2)))
                        pairs_to_process.append(pair)
        
        self.logger.info(f"Pre-filtering: {len(pairs_to_process)}/{total_pairs} pairs "
                        f"passed Jaccard threshold {threshold}")
        
        return pairs_to_process
    
    def _extract_fields_from_description(self, description: str) -> List[str]:
        """Extract field names from table description"""
        import re
        
        # Look for pattern "field_name (type)"
        pattern = r'(\w+)\s*\([^)]+\)'
        matches = re.findall(pattern, description)
        return [match.strip() for match in matches]

class ClusteringManager:
    """Main clustering manager that coordinates different clustering algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.algorithms = {
            'kmeans_manual': KMeansClusterer(),
            'kmeans_auto': KMeansClusterer(),
            'hierarchical': HierarchicalClusterer(),
            'dbscan_manual': DBSCANClusterer(),
            'dbscan_auto': DBSCANClusterer()
        }
        self.merger = ClusterMerger()
    
    def cluster_tables(self, features: np.ndarray, table_names: List[str],
                      method: str, param: Optional[any] = None) -> List[List[str]]:
        """
        Perform clustering using specified method and parameters
        """
        if method not in self.algorithms:
            raise ValueError(f"Unknown clustering method: {method}")
        
        algorithm = self.algorithms[method]
        clusters = algorithm.fit_predict(features, table_names, param)
        
        self.logger.info(f"Clustering with {method} (param: {param}) resulted in {len(clusters)} clusters")
        return clusters
    
    def run_all_clustering_methods(self, features: np.ndarray, table_names: List[str]) -> Dict[str, List[List[str]]]:
        """
        Run all configured clustering methods
        """
        results = {}
        
        for method, method_config in config.clustering.clustering_methods.items():
            for param in method_config['params']:
                key = f"{method}_{param}" if param is not None else method
                try:
                    clusters = self.cluster_tables(features, table_names, method, param)
                    results[key] = clusters
                except Exception as e:
                    self.logger.error(f"Error running {method} with param {param}: {e}")
        
        return results
    
    def evaluate_clustering_quality(self, features: np.ndarray, clusters: List[List[str]],
                                  table_names: List[str]) -> Dict[str, float]:
        """
        Evaluate clustering quality using various metrics
        """
        # Convert clusters to labels
        labels = np.zeros(len(table_names))
        for cluster_id, cluster in enumerate(clusters):
            for table in cluster:
                if table in table_names:
                    table_idx = table_names.index(table)
                    labels[table_idx] = cluster_id
        
        metrics = {}
        
        try:
            if len(set(labels)) > 1:
                silhouette = silhouette_score(features, labels)
                metrics['silhouette_score'] = silhouette
            else:
                metrics['silhouette_score'] = 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating silhouette score: {e}")
            metrics['silhouette_score'] = 0.0
        
        metrics['num_clusters'] = len(clusters)
        metrics['largest_cluster_size'] = max(len(cluster) for cluster in clusters) if clusters else 0
        metrics['smallest_cluster_size'] = min(len(cluster) for cluster in clusters) if clusters else 0
        
        return metrics