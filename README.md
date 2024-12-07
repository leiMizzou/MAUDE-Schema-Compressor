# Maude Schema Compressor

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Evaluation](#evaluation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

Maude Schema Compressor is a comprehensive tool designed to analyze, cluster, and optimize database schemas. It connects to a PostgreSQL database, extracts table structures and sample data, generates descriptive summaries, and clusters similar tables using K-Means, Hierarchical Clustering, and DBSCAN. The tool further refines these clusters based on similarity scores obtained from the DeepSeek API, merges related tables, and provides detailed token count comparisons to assist in optimizing schema representations for various applications, including AI-driven analysis.

Additionally, the tool supports evaluation of clustering performance using various metrics, allowing users to assess the effectiveness of different clustering strategies and parameters.

Preprint Manuscript at https://www.medrxiv.org/content/10.1101/2024.12.03.24318439v1

## Features

- **Database Connectivity**: Connects to PostgreSQL databases to extract table structures and sample data.
- **Data Extraction & Anonymization**: Retrieves and redacts sensitive fields in sample data to ensure privacy.
- **Table Description Generation**: Creates comprehensive textual descriptions of each table, detailing structure and sample entries.
- **Feature Extraction**:
  - **TF-IDF Vectorization**: Converts table descriptions into numerical feature vectors suitable for clustering.
  - **SentenceTransformer Embeddings**: Generates semantic embeddings for table descriptions using pre-trained models.
- **Clustering Methods**:
  - **K-Means Clustering**: Supports both manual (fixed number of clusters) and automatic (optimal cluster detection) approaches.
  - **Hierarchical Clustering**: Performs agglomerative clustering with adjustable distance thresholds.
  - **DBSCAN Clustering**: Implements both manual parameter specification and automatic parameter selection for density-based clustering.
- **Similarity Scoring**: Utilizes the DeepSeek API to compute similarity scores between table pairs, with pre-filtering based on Jaccard similarity to optimize API usage.
- **Caching Mechanism**: Caches similarity scores to minimize redundant API calls and enhance performance.
- **Parallel Processing**: Employs multi-threading to efficiently handle API calls and similarity computations.
- **Evaluation Metrics**: Calculates Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Precision, Recall, and F1 Score to evaluate clustering performance against manual groupings.
- **Dimensionality Reduction**: Offers PCA-based dimensionality reduction to enhance feature processing.
- **Token Counting & Comparison**: Analyzes and compares token counts in schema representations before and after merging to aid in optimization and efficiency assessments.
- **Prompt File Generation**: Creates prompt files before and after merging, integrating contextual information for downstream applications.
- **Comprehensive Logging**: Provides detailed logs to monitor progress, debug issues, and track operations.
- **Resume Capability**: Supports resuming analysis from the last checkpoint in case of interruptions.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **PostgreSQL Database**
- **Git**
- **Jupyter Notebook** (since the code is in `main.ipynb`)

### Clone the Repository

```bash
git clone https://github.com/leiMizzou/maude-schema-analysis.git
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
SCHEMA=public  # Change to your schema if different

# Output Configuration
OUTPUT_DIR=maude_schema_analysis
POST_MERGED_DIR=maude_schema_merged
SIMILARITY_CACHE_FILE=similarity_cache.json
EVALUATION_RESULTS_FILE=evaluation_results.csv
MANUAL_GROUPING_FILE=manual_grouping.json

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Clustering and Feature Extraction Parameters
SIMILARITY_THRESHOLDS=0.7,0.8,0.9             # Similarity thresholds for merging
PREFILTER_JACCARD_THRESHOLD=0.1               # Pre-filtering Jaccard similarity threshold
DISTANCE_THRESHOLDS=1.0                       # Distance thresholds for hierarchical clustering
CLUSTERING_METHODS=kmeans_manual,hierarchical,dbscan_manual  # Clustering methods to use
KMEANS_CLUSTERS=3                             # Number of clusters for K-Means
FEATURE_EXTRACTION_METHODS=tfidf,sentence_transformer  # Feature extraction methods

# Random Seed and Encoding
RANDOM_SEED=42                                # Seed for reproducibility
TOKEN_ENCODING=gpt2                           # Token encoding method compatible with OpenAI's GPT models

# Context and Prompt Files
CONTEXT_FILE=context.txt                      # File containing global context
PROMPT_BEFORE_FILE=prompt_before_merging.txt  # Output file for prompts before merging
PROMPT_AFTER_FILE=prompt_after_merging.txt    # Output file for prompts after merging

# Additional Settings
SAMPLE_SIZE=3                                 # Number of sample data per table
RETRY_LIMIT=2                                 # Number of retries for failed API calls
OVERWRITE_EXISTING=True                       # Whether to overwrite existing fields
MAX_TOKENS=100000                             # Maximum token limit, adjust based on actual API
ENABLE_PCA=True                               # Whether to enable PCA dimensionality reduction
ENABLE_SIMILARITY_CALCULATION=True            # Whether to enable similarity calculation and merging
```

*Ensure to replace placeholder values (e.g., `your_database_name`, `your_deepseek_api_key`) with actual values.*

## Usage

Since both the main analysis code and the evaluation code are placed inside `main.ipynb`, you can run the entire workflow through this Jupyter Notebook.

### Steps to Run the Analysis:

1. **Open Jupyter Notebook**

   ```bash
   jupyter notebook main.ipynb
   ```

2. **Configure Environment Variables**

   Ensure that the `.env` file is properly configured with your database credentials, API keys, and other settings.

3. **Run the Notebook**

   Execute all cells in `main.ipynb` sequentially. The notebook is organized to perform the following steps:

   - **Data Extraction and Anonymization**: Connects to the database, extracts table structures and sample data, anonymizes sensitive information, and saves the data as JSON files in `OUTPUT_DIR`.
   - **Feature Extraction**: Generates table descriptions and extracts features using TF-IDF or SentenceTransformer embeddings.
   - **Clustering**: Performs clustering using K-Means, Hierarchical Clustering, and DBSCAN based on the configurations.
   - **Similarity Scoring and Merging**: Computes similarity scores using the DeepSeek API and merges clusters based on similarity thresholds.
   - **Evaluation**: Compares AI-generated groupings with manual groupings and calculates evaluation metrics (ARI, NMI, Precision, Recall, F1 Score).
   - **Merged Schema Generation**: Merges structures and samples of similar tables and saves them in `POST_MERGED_DIR`.
   - **Token Counting and Comparison**: Counts tokens before and after merging to assess the impact on schema representation size.
   - **Prompt Generation**: Generates prompt files before and after merging, integrating contextual information.
   - **Logging and Resume Capability**: Provides detailed logs and supports resuming from the last checkpoint.

### Note on API Usage

- Ensure that you have a valid **DeepSeek API Key** and that the API usage complies with your subscription limits.
- The script includes caching mechanisms to minimize redundant API calls.

## Directory Structure

```
maude-schema-analysis/
├── .env
├── main.ipynb
├── requirements.txt
├── maude_schema_analysis/
│   ├── <table_name>.json
│   └── ... other table JSON files ...
├── maude_schema_merged/
│   ├── Merged_Table_1.json
│   ├── Merged_Table_2.json
│   └── ... other merged table JSON files ...
├── similarity_cache.json
├── manual_grouping.json
├── evaluation_results.csv
├── context.txt                  # Optional, if using global context
├── prompt_before_merging.txt
├── prompt_after_merging.txt
├── README.md
└── LICENSE
```

### Folder and File Descriptions

- **`.env`**: Environment variables configuration file. Contains settings such as database credentials, API keys, clustering parameters, and file paths.

- **`main.ipynb`**: The main Jupyter Notebook containing both the analysis and evaluation code. It orchestrates the entire workflow, from data extraction to evaluation and token comparison.

- **`requirements.txt`**: Lists all Python dependencies required to run the project.

- **`maude_schema_analysis/`**: Directory specified by `OUTPUT_DIR` in the `.env` file. Contains JSON files for each table after data extraction and anonymization.

  - **`<table_name>.json`**: JSON files for each table containing table structure and anonymized sample data.

- **`maude_schema_merged/`**: Directory specified by `POST_MERGED_DIR` in the `.env` file. Contains JSON files for each merged table.

  - **`Merged_Table_1.json`**, **`Merged_Table_2.json`**, etc.: JSON files representing merged tables, combining structures and sample data from similar tables.

- **`similarity_cache.json`**: Cache file storing computed similarity scores between table pairs to avoid redundant API calls.

- **`manual_grouping.json`**: JSON file containing manual grouping results. If not present, it is initialized using AI-generated groupings. Serves as the ground truth for evaluating AI clustering performance.

- **`evaluation_results.csv`**: CSV file recording evaluation metrics comparing AI-generated groupings with manual groupings. Includes metrics such as Adjusted Rand Index, Normalized Mutual Information, Precision, Recall, and F1 Score.

- **`context.txt`**: Optional. File containing global context information to be included in prompt generation.

- **`prompt_before_merging.txt`**: Generated prompt file containing context and descriptions of tables before merging. Used for AI-driven analysis or downstream applications.

- **`prompt_after_merging.txt`**: Generated prompt file containing context and descriptions of tables after merging. Reflects the optimized schema post-clustering and merging.

- **`README.md`**: Project documentation file (this file).

- **`LICENSE`**: License file detailing the project's licensing information.

## Evaluation

Maude Schema Compressor provides robust evaluation metrics to assess the quality and effectiveness of clustering results. It compares AI-generated groupings with manual (expert-based) groupings and calculates the following metrics:

- **Adjusted Rand Index (ARI)**: Measures the similarity between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

- **Normalized Mutual Information (NMI)**: Evaluates the agreement between two clusterings by measuring the mutual information normalized against the average entropy of the clusterings.

- **Precision, Recall, and F1 Score**: Assess the accuracy of the clustering by evaluating the correctness of group assignments in terms of true positives, false positives, and false negatives.

### How to Interpret Evaluation Metrics

- **ARI and NMI**: Higher values (closer to 1) indicate better agreement between AI-generated and manual groupings. Values near 0 suggest random labeling.

- **Precision and Recall**: Higher precision indicates fewer false positives, while higher recall indicates fewer false negatives.

- **F1 Score**: Balances precision and recall, providing a single metric to evaluate overall clustering accuracy.

### Viewing Evaluation Results

After running the analysis in `main.ipynb`, evaluation metrics are saved in the `evaluation_results.csv` file located in the project's root directory. You can open this file using any spreadsheet software or data analysis tool to review and interpret the results.

### Token Count Comparison

At the end of the analysis, the script outputs a comparison of token counts before and after merging, similar to the following:

```
=== Token Count Comparison ===
Total tokens in context.txt: 1500
Total tokens before merging (including context): 4500
Total tokens after merging (including context): 3000
Tokens from table descriptions before merging: 3000
Tokens from table descriptions after merging: 1500
Token reduction in table descriptions: 1500 tokens (50.00%)
```

This comparison helps assess the impact of merging on schema representation size, which is crucial for optimizing inputs for AI models with token limitations.

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

For any questions or suggestions, please open an issue on the [GitHub repository](https://github.com/leiMizzou/maude-schema-analysis/issues) or contact [lhua0420@gmail.com](mailto:lhua0420@gmail.com).

---

*This project is maintained by [Lei Hua](https://github.com/leiMizzou).*
