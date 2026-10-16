import ast
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="CemCat - Cementitious Materials Catalogue", layout="wide")

NER_CSV_PATH = Path("data/ner_cems_40k.csv")
ABSTRACT_JSON_PATH = Path("data/pii_abs.json")
PII_BASE_URL = "https://www.sciencedirect.com/science/article/pii/"
NER_TAG_COLUMNS = ["SMT", "APL", "MAT", "PRO", "DSC", "CMT", "SPL"]
WRITE_FONT_SIZE_PX = 30

DATASETS = {
    "Oxide Compositions": {
        "path": Path("upload/composition_cls_slag_updated.csv"),
        "compound_columns": ["SiO2", "Al2O3", "Fe2O3", "CaO", "Na2O", "K2O", "MgO", "SO3"],
        "extra_result_columns": ["mid", "total"],
        "description": (
            "Filter based on oxide compositions and NER tags."
        ),
    },
    "Mineralogical Compositions": {
        "path": Path("data/clinker_comps_with_dates.csv"),
        "compound_columns": ["C3S", "C2S", "C3A", "C4AF", "C4A3S"],
        "extra_result_columns": ["index", "date", "rcvd_year"],
        "description": (
            "Filter based on mineralogical compositions and NER tags."
        ),
    },
}


def stringify_tag_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if not text.strip():
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            cleaned = [str(item).strip() for item in parsed if str(item).strip()]
            return ", ".join(cleaned)
    except (ValueError, SyntaxError):
        pass
    return text


@st.cache_data(show_spinner=False)
def load_main_data(dataset_name: str) -> pd.DataFrame:
    config = DATASETS[dataset_name]
    df = pd.read_csv(config["path"], low_memory=False)

    if "pii" in df.columns:
        df["pii"] = df["pii"].fillna("").astype(str)
    else:
        df["pii"] = ""

    if "mid" in df.columns:
        mid_raw = df["mid"].where(df["mid"].notna(), "")
        mid_parts = mid_raw.astype(str).str.split("_")
        df["pii"] = df["pii"].where(df["pii"].ne(""), mid_parts.str[0])
        df["mid"] = mid_parts.str[-1]

    if "pii_tid" in df.columns:
        tid_last = pd.to_numeric(df["pii_tid"].astype(str).str.split("_").str[-1], errors="coerce")
        df["table_number"] = (tid_last + 1).astype("Int64")
        df["pii"] = df["pii"].where(df["pii"].ne(""), df["pii_tid"].astype(str).str.split("_").str[0])
    else:
        df["table_number"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    for col in config["compound_columns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_ner_data(path: Path) -> pd.DataFrame:
    ner_df = pd.read_csv(path)
    if "PII" not in ner_df.columns:
        ner_df["PII"] = ""
    ner_df["PII"] = ner_df["PII"].fillna("").astype(str)

    for col in NER_TAG_COLUMNS:
        if col not in ner_df.columns:
            ner_df[col] = ""
        ner_df[col] = ner_df[col].map(stringify_tag_cell)

    return ner_df[["PII"] + NER_TAG_COLUMNS].copy()


@st.cache_data(show_spinner=False)
def load_abstract_map(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def apply_composition_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, spec in filters.items():
        if spec["mode"] == "exact":
            mask &= df[col].notna() & (df[col] == spec["value"])
        else:
            mask &= df[col].notna() & (df[col] >= spec["min"]) & (df[col] <= spec["max"])
    return df[mask].copy()


def apply_ner_filter(ner_df: pd.DataFrame, ner_columns: list[str], keyword: str) -> pd.DataFrame:
    if not keyword.strip():
        return ner_df.copy()
    keyword_lower = keyword.lower()
    mask = pd.Series(False, index=ner_df.index)
    for col in ner_columns:
        if col in ner_df.columns:
            mask |= ner_df[col].fillna("").astype(str).str.lower().str.contains(keyword_lower, regex=False)
    return ner_df[mask].copy()


def write_large_text(text: str, font_size: int = WRITE_FONT_SIZE_PX) -> None:
    st.markdown(f'<p style="font-size: {font_size}px;">{text}</p>', unsafe_allow_html=True)


def format_compound_label(compound: str, dataset_name: str) -> str:
    # Oxides use Hill notation; mineralogical labels remain plain text.
    if dataset_name == "Oxide Compositions":
        return re.sub(r"(\d+)", r"<sub>\1</sub>", compound)
    if dataset_name == "Mineralogical Compositions" and compound == "C4A3S":
        return "C4A3<span style='text-decoration: overline;'>S</span>"
    return compound


def build_applied_filters_summary(
    dataset_name: str,
    comp_filters: dict,
    selected_ner_cols: list[str],
    ner_keyword: str,
) -> dict:
    comp_summary = {}
    for compound, spec in comp_filters.items():
        if spec["mode"] == "exact":
            comp_summary[compound] = {"mode": "Exact", "value": spec["value"]}
        else:
            comp_summary[compound] = {
                "mode": "Range",
                "min": spec["min"],
                "max": spec["max"],
            }
    return {
        "dataset": dataset_name,
        "composition": comp_summary,
        "ner": {"columns": selected_ner_cols, "keyword": ner_keyword.strip()},
    }


def main() -> None:
    st.title("CemCat")
    write_large_text("Cementitious Materials Catalogue", font_size=30)

    dataset_name = st.radio(
        "Dataset",
        options=list(DATASETS.keys()),
        horizontal=True,
    )
    config = DATASETS[dataset_name]
    dataset_key = "cement" if dataset_name == "Cement Composition" else "clinker"
    write_large_text(config["description"], font_size=20)

    main_df = load_main_data(dataset_name)
    ner_df = load_ner_data(NER_CSV_PATH)
    abstract_map = load_abstract_map(ABSTRACT_JSON_PATH)

    available_compounds = [col for col in config["compound_columns"] if col in main_df.columns]
    available_ner_cols = [col for col in NER_TAG_COLUMNS if col in ner_df.columns]

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("1) Composition Criteria")
        comp_filters: dict = {}
        compound_cols = st.columns(2)
        for idx, compound in enumerate(available_compounds):
            with compound_cols[idx % 2]:
                st.markdown(f"<b>{format_compound_label(compound, dataset_name)}</b>", unsafe_allow_html=True)
                enable = st.checkbox("Apply", key=f"{dataset_key}_use_{compound}")
                if not enable:
                    continue

                mode = st.radio(
                    "Mode",
                    options=["Range", "Exact"],
                    horizontal=True,
                    key=f"{dataset_key}_mode_{compound}",
                )
                series = main_df[compound].dropna()
                if series.empty:
                    st.warning(f"No numeric values for {compound}.")
                    continue

                if mode == "Exact":
                    exact_val = st.number_input(
                        "Exact",
                        value=float(series.median()),
                        key=f"{dataset_key}_exact_{compound}",
                    )
                    comp_filters[compound] = {"mode": "exact", "value": float(exact_val)}
                else:
                    min_val = st.number_input(
                        "Min",
                        value=float(series.min()),
                        key=f"{dataset_key}_min_{compound}",
                    )
                    max_val = st.number_input(
                        "Max",
                        value=float(series.max()),
                        key=f"{dataset_key}_max_{compound}",
                    )
                    if min_val > max_val:
                        st.error("Min cannot be greater than max.")
                        continue
                    comp_filters[compound] = {
                        "mode": "range",
                        "min": float(min_val),
                        "max": float(max_val),
                    }

        st.subheader("2) NER Criteria")
        selected_ner_cols = st.multiselect(
            "NER tag columns to search",
            options=available_ner_cols,
            default=available_ner_cols,
            key=f"{dataset_key}_ner_cols",
        )
        ner_keyword = st.text_input(
            "Search text for NER tags",
            placeholder="e.g., slag, hydration, porosity",
            key=f"{dataset_key}_ner_keyword",
        )

    filtered_comp = apply_composition_filters(main_df, comp_filters) if comp_filters else main_df.copy()
    filtered_ner = apply_ner_filter(ner_df, selected_ner_cols, ner_keyword) if selected_ner_cols else ner_df.iloc[0:0]

    result_df = filtered_comp.merge(filtered_ner, left_on="pii", right_on="PII", how="inner")
    result_df["abstract"] = result_df["pii"].map(abstract_map).fillna("")
    result_df["pii"] = result_df["pii"].fillna("").astype(str).map(
        lambda pii: f"{PII_BASE_URL}{pii}" if pii else ""
    )

    result_columns = (
        ["pii", "table_number"]
        + available_compounds
        + [col for col in config["extra_result_columns"] if col in result_df.columns]
        + available_ner_cols
        + ["abstract"]
    )
    result_columns = [col for col in result_columns if col in result_df.columns]
    final_df = result_df[result_columns].copy()

    with right_col:
        st.subheader("Results")
        st.caption(
            f"Composition matches: {len(filtered_comp)} rows | "
            f"NER matches: {filtered_ner['PII'].nunique()} papers | "
            f"Combined matches: {len(final_df)} rows"
        )
        st.markdown("**Applied Filters**")
        st.json(build_applied_filters_summary(dataset_name, comp_filters, selected_ner_cols, ner_keyword))
        st.dataframe(
            final_df,
            use_container_width=True,
            column_config={
                "pii": st.column_config.LinkColumn(
                    "pii",
                    help="Open the article page on ScienceDirect",
                    display_text=rf"{PII_BASE_URL}(.*)",
                ),
                "abstract": st.column_config.TextColumn("abstract", width="large"),
            },
        )
        st.download_button(
            label="Download combined results as CSV",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset_key}_ner_combined_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
