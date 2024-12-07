# MAUDE Database Analysis and Integration

This repository contains files and resources related to the analysis and integration of the **MAUDE** (Manufacturer and User Facility Device Experience) database, a U.S. FDA-maintained dataset for medical device adverse event reports.

## Overview
The MAUDE database includes detailed records of medical device failures, patient problems, and adverse events. This repository provides scripts that automate database integration, querying, and analysis to facilitate the identification of trends and critical issues related to medical device safety.

## Problem Statement
Medical device safety is a critical aspect of healthcare. Adverse events and device malfunctions can have serious implications for patient health and outcomes. The MAUDE database provides a wealth of information, but its size and complexity pose significant challenges for efficient analysis and actionable insights.

## Solution
This repository addresses these challenges by offering a streamlined approach to:
- **Data Integration**: Merging and harmonizing multiple years of MAUDE data into cohesive tables.
- **Automated Querying**: SQL scripts to extract relevant information quickly.
- **Insights Generation**: Tools to analyze trends, identify high-risk devices, and summarize adverse events.

## Research Objective
The primary objective of this repository is to streamline the analysis of the MAUDE database by automating key aspects of data processing, querying, and visualization. The repository ensures that research questions and SQL queries are dynamically generated to address specific analytical goals, eliminating manual overhead and enhancing reproducibility.

## Automated Workflow

The code in this repository automates the following key workflows:

### 1. **Research Question Formulation**
   - Users input their research goals or analytical focus (e.g., "Identify high-risk devices," "Analyze trends in adverse events").
   - The program automatically generates specific research questions based on user input and data schema.

### 2. **SQL Query Generation**
   - For each research question, SQL code is dynamically generated to extract relevant data.
   - Queries account for complex relationships across multiple merged tables (e.g., linking patient data with device event data).
   - Examples include:
     - Fetching adverse events by device type.
     - Analyzing event frequency trends over time.
     - Identifying correlations between patient demographics and adverse outcomes.

### 3. **Data Processing and Integration**
   - Raw MAUDE datasets are cleaned, validated, and merged into comprehensive tables.
   - Data integration scripts handle inconsistencies and missing values across multiple years of reports.

### 4. **Automated Analysis and Visualization**
   - Predefined Python scripts execute SQL queries, process results, and generate actionable insights.
   - Visualizations such as trend graphs, heatmaps, and risk matrices are automatically created to highlight key findings.

### 5. **Reporting and Documentation**
   - Findings are summarized into structured reports with visual and textual narratives.
   - Reports emphasize actionable insights and are dynamically tailored to the research objective.

## Contents

### Files
- **`sql.ipynb`**: Jupyter notebook containing SQL queries and analysis of the MAUDE database. The notebook demonstrates dynamic query generation and analysis workflows.

- **`prompt.txt`**: Text file documenting the structure and description of fields in the MAUDE database and the merged tables. This file is a key reference for understanding the data schema and metadata.

### Key Tables and Fields
The repository integrates multiple MAUDE data tables, providing a unified view of medical device events:

#### 1. **Merged_Table_1**
  - **Fields**: Includes `exemptn_no`, `mfr_name`, `report_id`, `event_type`, `brand_name`, `product_code`, and more.
  - **Purpose**: Combines records from multiple years (e.g., ASR_2007, ASR_2017) to provide a longitudinal view of device-related events.

#### 2. **Merged_Table_3**
  - **Fields**: Includes `exemption_number`, `manufacturer_name`, `report_id`, `event_type`, `device_problem_codes`, etc.
  - **Purpose**: Focuses on event reports from 2019, detailing adverse events and device problems.

#### 3. **Merged_Table_7**
  - **Fields**: Includes `mdr_report_key`, `report_number`, `event_type`, `date_received`, `adverse_event_flag`, etc.
  - **Purpose**: Provides detailed metadata on reports, including adverse events and product problem flags.

#### 4. **Merged_Table_8**
  - **Fields**: Includes `mdr_report_key`, `patient_sequence_number`, `patient_age`, `patient_sex`, and more.
  - **Purpose**: Consolidates patient data and treatment outcomes from various years.

#### 5. **Merged_Table_9**
  - **Fields**: Includes `mdr_report_key`, `foi_text`, `foi_text_json`.
  - **Purpose**: Captures free-text descriptions of adverse events, providing narrative context.

### Field Descriptions
The `prompt.txt` file includes comprehensive descriptions of all fields within the MAUDE database, covering topics such as:
- **Device Information**: Model numbers, product codes, and device problem codes.
- **Event Information**: Types of events, dates, and locations.
- **Reporter Information**: Reporter occupation, health professional status, and country codes.
- **Patient Information**: Age, sex, weight, ethnicity, and race.

### Example Data
Sample rows for key tables are provided in the `prompt.txt` file, illustrating the structure and content of merged data.

## Usage

### Prerequisites
- Install Python and Jupyter Notebook to execute the analysis scripts.
- A database client capable of handling SQL queries, such as SQLite or PostgreSQL.

### Running the Analysis

#### 1. Data Loading
   - Open the `sql.ipynb` notebook.
   - Load raw MAUDE datasets into the database.
   - Scripts in the notebook will assist in cleaning and transforming the data.

#### 2. Data Integration
   - Use the provided SQL scripts to merge datasets from multiple years into unified tables (e.g., `Merged_Table_1` through `Merged_Table_9`).
   - Validate the integrity of merged data using key identifiers such as `report_id` and `mdr_report_key`.

#### 3. Query Execution
   - Leverage dynamically generated SQL queries tailored to specific research questions.
   - Extract data subsets based on user-defined objectives.

#### 4. Analysis and Visualization
   - Utilize Jupyter Notebook to run Python scripts that execute queries and process results.
   - Automatically generate visualizations (e.g., trend graphs, heatmaps) to identify high-risk devices or recurring problems.

#### 5. Reporting Insights
   - Summarize findings into structured reports.
   - Tailor results and visualizations to address the initial research question dynamically.

### Applications
This repository is intended for researchers and analysts working with medical device safety data. Common use cases include:
- Trend analysis of adverse events.
- Evaluation of device performance and safety.
- Identification of high-risk medical devices.

## Contributing
Contributions are welcome! If you have suggestions for improving the analysis or additional resources to share, please submit a pull request or open an issue.

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the contents for your research or applications.

## Acknowledgments
This repository leverages publicly available data from the FDA's MAUDE database. The analysis scripts and documentation aim to facilitate understanding and usability of this critical dataset.

---

For more details on the MAUDE database, refer to the [FDA MAUDE Database Documentation](https://www.fda.gov/medical-devices/medical-device-reporting-mdr/how-search-maude).


