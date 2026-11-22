# Healthcare Data Governance and Cleaning

**EdVyro Data Analytics Program – Task 2**

A privacy-aware healthcare data analytics project focused on data governance, data quality validation, reproducible cleaning, and responsible handling of de-identified patient records using Python and Pandas.

---

## Project Overview

Healthcare data quality is essential for reliable analysis and responsible data management. This project focuses on preparing de-identified patient records by identifying data quality issues, validating data fields, applying reproducible cleaning rules, and documenting exceptions.

The project follows a privacy-aware approach and avoids inventing or modifying clinical values without proper justification.

## Objectives

- Classify healthcare data fields based on privacy and sensitivity.
- Create a comprehensive data dictionary.
- Identify missing values and duplicate records.
- Validate data types and numeric ranges.
- Detect potential data quality issues in clinical fields.
- Apply reproducible data cleaning rules.
- Generate a cleaned dataset.
- Create a detailed exception log.
- Evaluate overall data quality.
- Document findings and recommendations.

## Dataset

**Source:** EdVyro Healthcare Patient Sample Dataset

Dataset Link: https://edvyro.in/datasets/healthcare_patients_sample.csv

The dataset contains de-identified sample patient records intended for educational data analytics purposes.

> This project uses the provided sample dataset for learning and analytical purposes. The dataset is not intended for clinical decision-making.

## Technologies Used

- Python
- Pandas
- NumPy
- Jupyter Notebook
- Matplotlib
- Seaborn
- Streamlit
- Plotly
- GitHub

## Project Structure

```text
Healthcare-Data-Governance-Cleaning/
│
├── app.py
├── data_processor.py
├── validation_rules.py
├── data_dictionary.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── healthcare_patients_sample.csv
│
├── notebooks/
│   └── healthcare_data_cleaning.ipynb
│
├── outputs/
│   ├── cleaned_healthcare_data.csv
│   ├── data_dictionary.csv
│   ├── exception_log.csv
│   └── data_quality_report.csv
│
├── reports/
│   └── data_quality_report.md
│
└── tests/
    ├── test_validation.py
    └── test_data_processor.py
```

## Data Governance

The project applies privacy-aware data governance principles.

### Data Classification

Healthcare data fields are categorized based on their characteristics:

- Direct identifiers
- Quasi-identifiers
- Clinical information
- Demographic information
- Administrative information
- Non-sensitive metadata

### Privacy Assessment

The dataset is reviewed for potential privacy risks, including direct identifiers and combinations of fields that could potentially identify individuals.

De-identification is assessed based on the available dataset and documentation rather than assumed.

## Data Quality Validation

The following validation checks are performed:

- Missing value detection
- Duplicate patient ID identification
- Duplicate record detection
- Data type validation
- Numeric range validation
- Invalid value detection
- Categorical consistency checks
- Empty column identification
- Structural validation

## Data Cleaning Methodology

The cleaning process follows reproducible rules:

1. Load the original dataset.
2. Preserve the original data.
3. Normalize column names.
4. Handle whitespace inconsistencies.
5. Validate data types.
6. Identify missing values.
7. Detect duplicate records.
8. Flag invalid values.
9. Apply appropriate cleaning rules.
10. Generate the cleaned dataset.
11. Generate a data dictionary.
12. Generate an exception log.

Clinical values are not arbitrarily modified or replaced with invented values.

## Data Quality Metrics

The project evaluates data quality using the following dimensions:

| Metric | Description |
|---|---|
| Completeness | Measures the presence of required data |
| Validity | Checks whether values follow defined validation rules |
| Uniqueness | Identifies duplicate records and identifiers |
| Consistency | Detects inconsistent formats and categories |
| Structural Integrity | Evaluates dataset structure and data types |

An overall data quality score is calculated using documented rules and actual dataset results.

## Project Outputs

| Output File | Description |
|---|---|
| `cleaned_healthcare_data.csv` | Validated and cleaned patient dataset |
| `data_dictionary.csv` | Field descriptions, types, classifications, and validation rules |
| `exception_log.csv` | Identified data quality issues and cleaning actions |
| `data_quality_report.csv` | Summary of data quality metrics |
| `data_quality_report.md` | Detailed analytical report |

## Dashboard Features

The interactive dashboard provides:

- Dataset overview
- Data quality KPIs
- Missing value analysis
- Duplicate record analysis
- Data type analysis
- Validation results
- Data dictionary
- Exception log
- Cleaning pipeline
- Privacy and governance assessment
- Downloadable outputs

## Installation

Clone the repository:

```bash
git clone https://github.com/sarathkumarraids2025/Healthcare-Data-Governance-Cleaning.git
```

Navigate to the project directory:

```bash
cd Healthcare-Data-Governance-Cleaning
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser.

## Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/healthcare_data_cleaning.ipynb
```

## Run Tests

Execute the automated tests:

```bash
pytest -v
```

## Key Learning Outcomes

- Healthcare data governance
- Privacy-aware data handling
- Data quality assessment
- Data validation
- Data cleaning using Pandas
- Data dictionary creation
- Exception logging
- Reproducible data processing
- Data quality reporting
- Python data analytics

## Limitations

- The dataset is intended for educational analysis.
- Data quality checks do not establish clinical accuracy.
- Validation ranges are configurable screening rules.
- Missing clinical information is not artificially generated.
- The analysis does not provide medical diagnosis or treatment recommendations.

## Disclaimer

This project is developed for educational and data analytics purposes as part of the EdVyro Data Analytics Program.

It does not provide medical diagnosis, treatment recommendations, or clinical decision-making support. Clinical data validation rules are used only for data quality assessment.

## Author

**Sarathkumar R**

B.Tech – Artificial Intelligence and Data Science

Data Analytics | Python | SQL | Machine Learning | Data Visualization

GitHub: [sarathkumarraids2025](https://github.com/sarathkumarraids2025)

---

*EdVyro Data Analytics Program – Task 2: Healthcare Data Governance and Cleaning*
