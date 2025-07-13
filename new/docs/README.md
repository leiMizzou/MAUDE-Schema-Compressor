# MAUDE Schema Compressor - Standalone Package

This is a complete, independent, and concise standalone package for the MAUDE Schema Compressor project. It contains all necessary data and code to run the semantic clustering and merging framework (SCMF) without requiring database connections.

## Contents

- **Python modules**: Core implementation files
- **data/**: Raw table data (113 JSON files) and reference files
  - 113 table schema JSON files from MAUDE database
  - `manual_grouping.json`: Reference standard answers for evaluation
  - `similarity_cache.json`: Pre-computed similarity scores (1,425 cached)
- **analysis_outputs/**: Directory for generated results

## Key Features

- Semantic clustering using TF-IDF and Sentence Transformers
- Multiple clustering algorithms (K-Means, Hierarchical, DBSCAN)
- DeepSeek API integration for semantic similarity (with caching)
- Comprehensive evaluation metrics (ARI, NMI, F1-score)
- Standalone operation without database dependency

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python run_analysis.py

# Run clustering only (no API calls)
python standalone_runner.py --clustering-only

# Use cached similarities only
python standalone_runner.py --no-api
```

## Analysis Results

The framework achieved excellent performance with F1 scores up to 1.0, demonstrating effective schema clustering and merging capabilities. Results show that combining clustering with API-based semantic similarity significantly improves accuracy over clustering-only approaches.

## Configuration

Edit `.env` file to customize:
- API keys and settings
- File paths and directories
- Processing parameters

All paths are configured for standalone operation with relative paths.