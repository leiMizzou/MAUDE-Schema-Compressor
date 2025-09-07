"""
Database module for MAUDE Schema Compressor
Handles database connections, data extraction, and table information management
"""
import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, inspect, text, Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from tqdm import tqdm

from config import config

class DatabaseManager:
    """Manages database connections and data extraction"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.inspector = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Establish database connection
        Returns True if connection successful, False otherwise
        """
        try:
            self.engine = create_engine(config.get_database_uri())
            
            # Test connection
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            self.inspector = inspect(self.engine)
            self.logger.info("Successfully connected to the database")
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def get_all_tables(self) -> List[str]:
        """
        Get all table names in the specified schema
        Returns list of table names
        """
        if not self.inspector:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            tables = self.inspector.get_table_names(schema=config.database.schema)
            self.logger.info(f"Found {len(tables)} tables in schema '{config.database.schema}'")
            return tables
        except SQLAlchemyError as e:
            self.logger.error(f"Error retrieving table names: {e}")
            return []
    
    def get_table_structure(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get structure information for a specific table
        Returns list of column information dictionaries
        """
        if not self.inspector:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            columns = self.inspector.get_columns(table_name, schema=config.database.schema)
            column_info = []
            
            for column in columns:
                column_info.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': str(column['default']) if column['default'] else None
                })
            
            return column_info
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error retrieving table structure for {table_name}: {e}")
            return []
    
    def get_sample_data(self, table_name: str, sample_size: int = None) -> List[Dict[str, Any]]:
        """
        Get sample data from a specific table
        Returns list of sample records
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        if sample_size is None:
            sample_size = config.processing.sample_size
        
        try:
            query = f'SELECT * FROM "{config.database.schema}"."{table_name}" LIMIT {sample_size};'
            df = pd.read_sql(query, self.engine)
            return df.to_dict(orient='records')
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error retrieving sample data for {table_name}: {e}")
            return []
    
    def anonymize_sample_data(self, sample_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize sample data by redacting sensitive fields
        Returns anonymized data
        """
        sensitive_fields = [
            'patient_id', 'first_name', 'last_name', 'date_of_birth', 
            'name', 'address', 'phone', 'email', 'ssn'
        ]
        
        anonymized_data = []
        for record in sample_data:
            anonymized_record = {}
            for key, value in record.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    anonymized_record[key] = "REDACTED"
                else:
                    anonymized_record[key] = value
            anonymized_data.append(anonymized_record)
        
        return anonymized_data
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")

class TableDataManager:
    """Manages table data storage and retrieval"""
    
    def __init__(self, database_manager: DatabaseManager = None):
        self.db_manager = database_manager
        self.logger = logging.getLogger(__name__)
    
    def save_table_info(self, table_name: str, structure: List[Dict], samples: List[Dict]):
        """
        Save table structure and sample data to JSON file
        """
        # Anonymize sample data
        if self.db_manager:
            anonymized_samples = self.db_manager.anonymize_sample_data(samples)
        else:
            anonymized_samples = self._anonymize_sample_data_standalone(samples)
        
        data = {
            'table_name': table_name,
            'structure': structure,
            'sample_data': anonymized_samples
        }
        
        file_path = f"{config.paths.output_dir}/{table_name}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, default=str)
            
            # Validate JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            
            self.logger.info(f"Saved and validated data for table '{table_name}'")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON generated for table '{table_name}': {e}")
        except Exception as e:
            self.logger.error(f"Error saving table data for {table_name}: {e}")
    
    def _anonymize_sample_data_standalone(self, sample_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize sample data without database manager (standalone mode)
        """
        sensitive_fields = [
            'patient_id', 'first_name', 'last_name', 'date_of_birth', 
            'name', 'address', 'phone', 'email', 'ssn'
        ]
        
        anonymized_data = []
        for record in sample_data:
            anonymized_record = {}
            for key, value in record.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    anonymized_record[key] = "REDACTED"
                else:
                    anonymized_record[key] = value
            anonymized_data.append(anonymized_record)
        
        return anonymized_data
    
    def load_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Load table information from JSON file
        Returns table data or None if not found
        """
        file_path = f"{config.paths.output_dir}/{table_name}.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Table data file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error in {file_path}: {e}")
            return None
    
    def load_all_table_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all table data from JSON files
        Returns dictionary with table names as keys
        """
        import os
        
        data = {}
        if not os.path.isdir(config.paths.output_dir):
            self.logger.error(f"Output directory '{config.paths.output_dir}' does not exist")
            return data
        
        # Skip non-table JSON files
        excluded_files = {
            'manual_grouping.json', 'similarity_cache.json', 'experiment_summary.json',
            'maude_schema_analysis.json', 'evaluation_results.json'
        }
        
        json_files = [f for f in os.listdir(config.paths.output_dir) 
                     if f.endswith('.json') and f not in excluded_files]
        self.logger.info(f"Loading {len(json_files)} JSON files")
        
        for json_file in tqdm(json_files, desc="Loading table data"):
            file_path = f"{config.paths.output_dir}/{json_file}"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                
                # Check if it's a valid table data file
                if isinstance(table_data, dict) and 'table_name' in table_data:
                    table_name = table_data['table_name']
                    data[table_name] = table_data
                else:
                    # Skip files that don't have table structure
                    continue
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error in {json_file}: {e}")
            except Exception as e:
                self.logger.error(f"Error reading {json_file}: {e}")
        
        self.logger.info(f"Successfully loaded data for {len(data)} tables")
        return data
    
    def generate_table_description(self, table_name: str, structure: List[Dict], 
                                 samples: List[Dict]) -> str:
        """
        Generate textual description of table including fields and sample data
        """
        description = f"Table '{table_name}' has the following fields: "
        
        # Add field information
        fields = [f"{col['name']} ({col['type']})" for col in structure]
        description += ", ".join(fields) + ". "
        
        # Add sample data summary
        if samples:
            description += "Sample data includes: "
            sample_summaries = []
            
            for column in structure:
                col_name = column['name']
                # Extract sample values for each column
                sample_values = []
                for record in samples[:3]:  # First 3 samples
                    if col_name in record and record[col_name] is not None:
                        sample_values.append(str(record[col_name])[:50])  # Limit length
                
                if sample_values:
                    sample_summaries.append(f"{col_name} values like {', '.join(sample_values)}")
            
            description += "; ".join(sample_summaries) + "."
        
        return description
    
    def get_all_table_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all tables
        Returns dictionary with table names as keys and descriptions as values
        """
        descriptions = {}
        all_data = self.load_all_table_data()
        
        for table_name, table_data in all_data.items():
            structure = table_data.get('structure', [])
            samples = table_data.get('sample_data', [])
            description = self.generate_table_description(table_name, structure, samples)
            descriptions[table_name] = description
        
        self.logger.info(f"Generated descriptions for {len(descriptions)} tables")
        return descriptions
    
    def extract_and_save_all_tables(self) -> bool:
        """
        Extract and save all table information from database
        Returns True if successful, False otherwise
        """
        if not self.db_manager or not self.db_manager.engine:
            self.logger.error("Database not connected")
            return False
        
        try:
            tables = self.db_manager.get_all_tables()
            
            for table in tqdm(tables, desc="Processing tables"):
                structure = self.db_manager.get_table_structure(table)
                samples = self.db_manager.get_sample_data(table)
                
                if not structure:
                    self.logger.warning(f"Table '{table}' has empty structure")
                if not samples:
                    self.logger.warning(f"Table '{table}' has empty sample data")
                
                self.save_table_info(table, structure, samples)
            
            self.logger.info(f"Successfully processed {len(tables)} tables")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting table data: {e}")
            return False