"""
Table Schema Merger for MAUDE Schema Compressor
Handles merging of table schemas based on clustering results
"""
import json
import logging
import os
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
from config import config

class TableSchemaMerger:
    """Merges table schemas based on clustering results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge_table_schemas(self, clusters: List[List[str]], 
                           table_data: Dict[str, Dict],
                           experiment_info: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Merge table schemas within each cluster
        Returns dictionary of merged table schemas
        """
        merged_schemas = {}
        
        for cluster_id, cluster_tables in enumerate(clusters):
            if len(cluster_tables) == 1:
                # Single table - no merging needed
                table_name = cluster_tables[0]
                merged_schemas[f"Cluster_{cluster_id+1}_{table_name}"] = {
                    'original_tables': cluster_tables,
                    'merged_schema': table_data.get(table_name, {}),
                    'merge_type': 'single_table'
                }
            else:
                # Multiple tables - merge schemas
                merged_schema = self._merge_multiple_tables(cluster_tables, table_data)
                merged_schemas[f"Merged_Cluster_{cluster_id+1}"] = {
                    'original_tables': cluster_tables,
                    'merged_schema': merged_schema,
                    'merge_type': 'multi_table',
                    'table_count': len(cluster_tables)
                }
        
        self.logger.info(f"Created {len(merged_schemas)} merged schemas from {len(clusters)} clusters")
        return merged_schemas
    
    def _merge_multiple_tables(self, cluster_tables: List[str], 
                              table_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Merge multiple table schemas into a single consolidated schema
        """
        merged_schema = {
            'table_name': f"merged_{'_'.join(cluster_tables[:3])}{'_etc' if len(cluster_tables) > 3 else ''}",
            'source_tables': cluster_tables,
            'table_count': len(cluster_tables),
            'merged_fields': {},
            'field_statistics': {},
            'data_types': {},
            'constraints': {},
            'sample_data': {}
        }
        
        # Collect all fields from all tables
        all_fields = defaultdict(list)
        field_types = defaultdict(Counter)
        field_constraints = defaultdict(set)
        sample_values = defaultdict(list)
        
        for table_name in cluster_tables:
            table_info = table_data.get(table_name, {})
            structure = table_info.get('structure', [])
            sample_data = table_info.get('sample_data', [])
            
            # Process each field
            for field_info in structure:
                field_name = field_info.get('column_name', 'unknown')
                field_type = field_info.get('data_type', 'unknown')
                constraints = field_info.get('constraints', [])
                
                all_fields[field_name].append(table_name)
                field_types[field_name][field_type] += 1
                if constraints:
                    field_constraints[field_name].update(constraints)
            
            # Collect sample data for common fields
            if sample_data:
                for row in sample_data[:3]:  # Take first 3 rows
                    for field_name, value in row.items():
                        if value is not None and str(value).strip():
                            sample_values[field_name].append(str(value)[:50])  # Truncate long values
        
        # Merge field definitions
        for field_name, source_tables in all_fields.items():
            frequency = len(source_tables)
            coverage = frequency / len(cluster_tables)
            
            # Get most common data type
            most_common_type = field_types[field_name].most_common(1)[0][0]
            
            merged_schema['merged_fields'][field_name] = {
                'data_type': most_common_type,
                'source_tables': source_tables,
                'frequency': frequency,
                'coverage_percentage': round(coverage * 100, 2),
                'type_variations': dict(field_types[field_name]),
                'constraints': list(field_constraints[field_name]) if field_constraints[field_name] else [],
                'sample_values': list(set(sample_values[field_name][:5]))  # Top 5 unique samples
            }
        
        # Calculate field statistics
        merged_schema['field_statistics'] = {
            'total_unique_fields': len(all_fields),
            'common_fields': len([f for f, tables in all_fields.items() if len(tables) == len(cluster_tables)]),
            'frequent_fields': len([f for f, tables in all_fields.items() if len(tables) >= len(cluster_tables) * 0.5]),
            'rare_fields': len([f for f, tables in all_fields.items() if len(tables) == 1])
        }
        
        # Data type distribution
        all_types = []
        for type_counter in field_types.values():
            all_types.extend(type_counter.keys())
        merged_schema['data_types'] = dict(Counter(all_types))
        
        return merged_schema
    
    def save_merged_schemas(self, merged_schemas: Dict[str, Dict], 
                           experiment_info: Dict[str, Any]) -> str:
        """
        Save merged schemas to files
        """
        output_dir = config.paths.post_merged_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on experiment info
        method = experiment_info.get('clustering_method', 'unknown')
        param = experiment_info.get('clustering_param', 'none')
        sim_threshold = experiment_info.get('similarity_threshold')
        feature_method = experiment_info.get('feature_extraction_method', 'unknown')
        
        if sim_threshold is not None:
            filename = f"merged_schemas_{method}_{feature_method}_sim{sim_threshold}_param{param}.json"
        else:
            filename = f"merged_schemas_{method}_{feature_method}_param{param}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # Prepare output data
        output_data = {
            'experiment_info': experiment_info,
            'merge_summary': {
                'total_clusters': len(merged_schemas),
                'single_table_clusters': len([s for s in merged_schemas.values() if s['merge_type'] == 'single_table']),
                'multi_table_clusters': len([s for s in merged_schemas.values() if s['merge_type'] == 'multi_table']),
                'total_original_tables': sum(len(s['original_tables']) for s in merged_schemas.values())
            },
            'merged_schemas': merged_schemas
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved merged schemas to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving merged schemas: {e}")
            return ""
    
    def create_merge_summary_report(self, merged_schemas: Dict[str, Dict], 
                                   experiment_info: Dict[str, Any]) -> str:
        """
        Create a human-readable summary report of the merge results
        """
        output_dir = config.paths.output_dir
        
        method = experiment_info.get('clustering_method', 'unknown')
        param = experiment_info.get('clustering_param', 'none')
        sim_threshold = experiment_info.get('similarity_threshold')
        
        if sim_threshold is not None:
            filename = f"merge_summary_{method}_sim{sim_threshold}_param{param}.txt"
        else:
            filename = f"merge_summary_{method}_param{param}.txt"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("MAUDE Schema Merge Summary Report\n")
                f.write("=" * 80 + "\n\n")
                
                # Experiment info
                f.write("Experiment Configuration:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Clustering Method: {experiment_info.get('clustering_method', 'N/A')}\n")
                f.write(f"Parameters: {experiment_info.get('clustering_param', 'N/A')}\n")
                f.write(f"Feature Extraction: {experiment_info.get('feature_extraction_method', 'N/A')}\n")
                f.write(f"Similarity Threshold: {experiment_info.get('similarity_threshold', 'N/A')}\n")
                f.write(f"Final Clusters: {len(merged_schemas)}\n\n")
                
                # Overall statistics
                multi_table_schemas = [s for s in merged_schemas.values() if s['merge_type'] == 'multi_table']
                total_original_tables = sum(len(s['original_tables']) for s in merged_schemas.values())
                
                f.write("Merge Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Original Tables: {total_original_tables}\n")
                f.write(f"Total Merged Schemas: {len(merged_schemas)}\n")
                f.write(f"Single Table Schemas: {len(merged_schemas) - len(multi_table_schemas)}\n")
                f.write(f"Multi-Table Merges: {len(multi_table_schemas)}\n")
                f.write(f"Compression Ratio: {round(total_original_tables / len(merged_schemas), 2)}:1\n\n")
                
                # Detailed cluster information
                f.write("Detailed Cluster Analysis:\n")
                f.write("-" * 30 + "\n\n")
                
                for schema_name, schema_info in merged_schemas.items():
                    f.write(f"📊 {schema_name}\n")
                    f.write(f"   Type: {schema_info['merge_type']}\n")
                    f.write(f"   Original Tables ({len(schema_info['original_tables'])}): {', '.join(schema_info['original_tables'][:5])}")
                    if len(schema_info['original_tables']) > 5:
                        f.write(f" ... and {len(schema_info['original_tables']) - 5} more")
                    f.write("\n")
                    
                    if schema_info['merge_type'] == 'multi_table':
                        merged_schema = schema_info['merged_schema']
                        stats = merged_schema.get('field_statistics', {})
                        f.write(f"   Total Fields: {stats.get('total_unique_fields', 0)}\n")
                        f.write(f"   Common Fields: {stats.get('common_fields', 0)}\n")
                        f.write(f"   Frequent Fields: {stats.get('frequent_fields', 0)}\n")
                        f.write(f"   Rare Fields: {stats.get('rare_fields', 0)}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Created merge summary report: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error creating merge summary: {e}")
            return ""
    
    def get_best_merge_candidates(self, all_results: List[Dict]) -> List[Dict]:
        """
        Identify the best clustering results for schema merging
        """
        # Filter for high-performing experiments
        high_performance = [
            result for result in all_results 
            if result.get('f1_score', 0) > 0.9 and result.get('adjusted_rand_index', 0) > 0.9
        ]
        
        if not high_performance:
            # Fallback to top performers
            sorted_results = sorted(all_results, 
                                  key=lambda x: (x.get('f1_score', 0) + x.get('adjusted_rand_index', 0)) / 2, 
                                  reverse=True)
            high_performance = sorted_results[:3]
        
        self.logger.info(f"Selected {len(high_performance)} best experiments for schema merging")
        return high_performance