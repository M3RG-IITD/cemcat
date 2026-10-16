# CemCat - Cementitious Materials Catalogue

This repository includes a Streamlit dashboard for searching cementitious material data using:

- **Composition filters** (exact value or range)
- **NER tag keyword filters** (`SMT`, `APL`, `MAT`, `PRO`, `DSC`, `CMT`, `SPL`)
- **Paper metadata** (`pii`, `table_number`, `mid`) and **abstract text**

## Dashboard File

- `streamlit_composition_ner_combined_dashboard.py`

## Data Sources

- Composition: `upload/composition_cls_slag_updated.csv`
- NER tags: `data/ner_cems_40k.csv`
- Abstracts: `data/pii_abs.json`

## How to Run

1. Install dependencies:
   - `pip install streamlit pandas`
2. Start the app from the repository root:
   - `streamlit run streamlit_composition_ner_combined_dashboard.py`
3. Open the local URL shown in terminal (usually `http://localhost:8501`).

## How to Use

1. **Set composition criteria**  
   Expand any compound (`SiO2`, `Al2O3`, `Fe2O3`, `CaO`, `Na2O`, `K2O`, `MgO`, `SO3`) and choose:
   - `Exact`: match one value
   - `Range`: match min-max interval

2. **Set NER criteria**  
   - Select one or more NER tag columns
   - Enter a keyword (for example: `slag`, `chloride`, `porosity`)

3. **View combined results (AND logic)**  
   Results include rows that satisfy both composition and NER criteria.

## Output Columns

- Clickable `pii` link to ScienceDirect
- `table_number` (derived from `pii_tid` as `1 + last underscore segment`)
- Compound columns
- `mid` (last underscore-separated token)
- `total`
- NER tag columns (`SMT`, `APL`, `MAT`, `PRO`, `DSC`, `CMT`, `SPL`)
- `abstract`

## Export

Use **Download combined results as CSV** in the dashboard to export the filtered table.
