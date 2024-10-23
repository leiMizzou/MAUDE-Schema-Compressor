# Maude Schema Analysis

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

Maude Schema Analysis is a comprehensive tool designed to analyze and cluster database schemas. It connects to a PostgreSQL database, extracts table structures and sample data, and performs clustering based on table descriptions using various feature extraction and clustering methods. The tool leverages the DeepSeek API for similarity scoring and provides detailed analysis and evaluation metrics to assist in data architecture and quality control.

## Features

- **Database Connectivity**: Connects to PostgreSQL databases to extract table structures and sample data.
- **Data Anonymization**: Redacts sensitive fields in sample data to ensure privacy.
- **Feature Extraction**: Supports TF-IDF and SentenceTransformer embeddings for table descriptions.
- **Clustering Methods**: Implements K-Means, Hierarchical Clustering, and DBSCAN with both manual and automatic parameter selection.
- **Similarity Scoring**: Utilizes the DeepSeek API to compute similarity scores between table pairs.
- **Evaluation Metrics**: Calculates Adjusted Rand Index, Normalized Mutual Information, Precision, Recall, and F1 Score to evaluate clustering performance.
- **Caching Mechanism**: Caches similarity scores to optimize API usage and reduce redundant computations.
- **Parallel Processing**: Employs multi-threading for efficient API calls and similarity computations.
- **Comprehensive Logging**: Provides detailed logs for monitoring and debugging.
- **Resume Capability**: Supports resuming analysis from the last checkpoint in case of interruptions.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **PostgreSQL Database**

### Clone the Repository

```bash
git clone https://github.com/yourusername/maude-schema-analysis.git
cd maude-schema-analysis
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not provided, you can install dependencies manually:*

```bash
pip install sqlalchemy pandas python-dotenv tqdm scikit-learn numpy tiktoken requests networkx sentence-transformers joblib matplotlib seaborn
```

## Configuration

The tool relies on environment variables for configuration. Create a `.env` file in the root directory of the project and populate it with the necessary variables.

### `.env` File

```env
# Database Configuration
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=5432

# Output Configuration
OUTPUT_DIR=maude_schema_analysis
COMBINED_OUTPUT_FILE=maude_schema_combined.txt
OUTPUT_ANALYSIS_FILE=maude_schema_analysis.json
EVALUATION_RESULTS_FILE=evaluation_results.csv

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Additional Settings
SAMPLE_SIZE=3
RETRY_LIMIT=2
OVERWRITE_EXISTING=True
MAX_TOKENS=100000
SIMILARITY_THRESHOLDS=0.7,0.8,0.9
PREFILTER_JACCARD_THRESHOLD=0.1
DISTANCE_THRESHOLDS=1.0
GLOBAL_CONTEXT_FILE=context.txt
USE_HIERARCHICAL_CLUSTERING=True
ENABLE_SIMILARITY_CALCULATION=True
MANUAL_GROUPING_FILE=manual_grouping.json
CACHE_FILE=similarity_cache.json
ENABLE_PCA=True
FEATURE_EXTRACTION_METHODS=tfidf,sentence_transformer
```

*Ensure to replace placeholder values (e.g., `your_database_name`, `your_deepseek_api_key`) with actual values.*

## Usage

Once installed and configured, you can run the analysis script using the following command:

```bash
python main.py
```

### Script Workflow

1. **Database Connection**: Connects to the specified PostgreSQL database.
2. **Data Extraction**: Retrieves table structures and sample data.
3. **Data Anonymization**: Redacts sensitive fields in the sample data.
4. **Feature Extraction**: Generates feature vectors using TF-IDF or SentenceTransformer.
5. **Clustering**: Applies selected clustering algorithms to group similar tables.
6. **Similarity Scoring**: Computes similarity scores between table pairs using the DeepSeek API.
7. **Merging Clusters**: Merges table clusters based on similarity thresholds.
8. **Evaluation**: Compares AI-generated groupings with manual groupings and calculates evaluation metrics.
9. **Output Generation**: Saves results, including JSON files, combined output, analysis results, cache files, and evaluation metrics.

## Directory Structure

```
maude-schema-analysis/
├── .env
├── main.py
├── requirements.txt
├── maude_schema_analysis/
│   ├── <table_name>.json
│   ├── maude_schema_combined.txt
│   ├── maude_schema_analysis.json
│   ├── similarity_cache.json
│   ├── manual_grouping.json
│   ├── evaluation_results.csv
│   ├── groupings_kmeans_manual_sim_0.7_param_3.csv
│   ├── groupings_hierarchical_sim_0.8_param_1.0.csv
│   └── ... other output files ...
├── context.txt
└── README.md
```

### Folder and File Descriptions

- **`.env`**: Environment variables configuration file. Contains sensitive information such as database credentials and API keys.
  
- **`main.py`**: The main Python script that performs the schema analysis, clustering, similarity scoring, and evaluation.

- **`requirements.txt`**: Lists all Python dependencies required to run the project.

- **`maude_schema_analysis/`**: Default output directory specified by `OUTPUT_DIR` in the `.env` file.
  
  - **`<table_name>.json`**: JSON files for each table containing table structure and anonymized sample data.
  
  - **`maude_schema_combined.txt`**: Integrated file containing merged table information and global context.
  
  - **`maude_schema_analysis.json`**: JSON file storing analysis results from the DeepSeek API.
  
  - **`similarity_cache.json`**: Cache file to store computed similarity scores between table pairs to avoid redundant API calls.
  
  - **`manual_grouping.json`**: JSON file containing manual grouping results. If not present, it is initialized using AI-generated groupings.
  
  - **`evaluation_results.csv`**: CSV file recording evaluation metrics comparing AI groupings with manual groupings.
  
  - **`groupings_*.csv`**: CSV files capturing different grouping results based on clustering methods and similarity thresholds. The filename includes details like clustering method, similarity threshold, and clustering parameters (e.g., `groupings_kmeans_manual_sim_0.7_param_3.csv`).

- **`context.txt`**: Global context file used to integrate merged table information into a single large file for comprehensive analysis.

- **`README.md`**: Project documentation file (this file).

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of this page to create a copy of the repository under your GitHub account.

2. **Clone the Forked Repository**

   ```bash
   git clone https://github.com/yourusername/maude-schema-analysis.git
   cd maude-schema-analysis
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Implement your feature or bug fix.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add your descriptive commit message"
   ```

6. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

   Navigate to the original repository and click "Compare & pull request" to submit your changes for review.

## Contact

For any questions or suggestions, please open an issue on the [GitHub repository](https://github.com/yourusername/maude-schema-analysis/issues) or contact [your.email@example.com](mailto:your.email@example.com).

---

*This project is maintained by [Your Name](https://github.com/yourusername).*
