import ast
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Composition + NER Dashboard", layout="wide")

COMPOSITION_CSV_PATH = Path("upload/composition_cls_slag_updated.csv")
NER_CSV_PATH = Path("data/ner_cems_40k.csv")
ABSTRACT_JSON_PATH = Path("data/pii_abs.json")
PII_BASE_URL = "https://www.sciencedirect.com/science/article/pii/"
COMPOUND_COLUMNS = ["SiO2", "Al2O3", "Fe2O3", "CaO", "Na2O", "K2O", "MgO", "SO3"]
NER_TAG_COLUMNS = ["SMT", "APL", "MAT", "PRO", "DSC", "CMT", "SPL"]


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
def load_composition_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "mid" in df.columns:
        mid_raw = df["mid"].where(df["mid"].notna(), "")
        mid_parts = mid_raw.astype(str).str.split("_")
        df["pii"] = mid_parts.str[0]
        df["mid"] = mid_parts.str[-1]
    else:
        df["pii"] = ""
        df["mid"] = ""

    if "pii_tid" in df.columns:
        last_part = pd.to_numeric(df["pii_tid"].astype(str).str.split("_").str[-1], errors="coerce")
        df["table_number"] = (last_part + 1).astype("Int64")
    else:
        df["table_number"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    for col in COMPOUND_COLUMNS:
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

    keep_cols = ["PII"] + NER_TAG_COLUMNS
    return ner_df[keep_cols].copy()


@st.cache_data(show_spinner=False)
def load_abstract_map(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def apply_composition_filters(df: pd.DataFrame, filter_specs: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, spec in filter_specs.items():
        if spec["mode"] == "exact":
            mask &= df[col].notna() & (df[col] == spec["value"])
        else:
            mask &= df[col].notna() & (df[col] >= spec["min"]) & (df[col] <= spec["max"])
    return df[mask].copy()


def apply_ner_filter(ner_df: pd.DataFrame, ner_columns: list[str], keyword: str) -> pd.DataFrame:
    if not keyword.strip():
        return ner_df.copy()
    keyword_lower = keyword.lower()
    search_mask = pd.Series(False, index=ner_df.index)
    for col in ner_columns:
        if col in ner_df.columns:
            search_mask |= ner_df[col].fillna("").astype(str).str.lower().str.contains(keyword_lower, regex=False)
    return ner_df[search_mask].copy()


def main() -> None:
    st.title("Composition + NER + Abstract Dashboard")
    st.write(
        "Search by composition filters or by NER tags, then view matching composition rows "
        "with paper abstracts."
    )

    comp_df = load_composition_data(COMPOSITION_CSV_PATH)
    ner_df = load_ner_data(NER_CSV_PATH)
    abstract_map = load_abstract_map(ABSTRACT_JSON_PATH)

    search_mode = st.radio("Search mode", ["Composition", "NER Tag"], horizontal=True)

    filtered_comp = comp_df.copy()
    filtered_ner = ner_df.copy()
    available_compounds = [col for col in COMPOUND_COLUMNS if col in comp_df.columns]

    if search_mode == "Composition":
        st.subheader("Composition Filters")
        composition_filters: dict = {}

        for compound in available_compounds:
            with st.expander(compound, expanded=False):
                enable = st.checkbox(f"Apply filter for {compound}", key=f"use_{compound}")
                if not enable:
                    continue

                mode = st.radio(
                    f"Filter mode for {compound}",
                    options=["Range", "Exact"],
                    horizontal=True,
                    key=f"mode_{compound}",
                )

                series = comp_df[compound].dropna()
                if series.empty:
                    st.warning(f"No numeric values available for {compound}.")
                    continue

                if mode == "Exact":
                    exact_val = st.number_input(
                        f"Exact value ({compound})",
                        value=float(series.median()),
                        key=f"exact_{compound}",
                    )
                    composition_filters[compound] = {"mode": "exact", "value": float(exact_val)}
                else:
                    c1, c2 = st.columns(2)
                    min_val = c1.number_input(
                        f"Min {compound}",
                        value=float(series.min()),
                        key=f"min_{compound}",
                    )
                    max_val = c2.number_input(
                        f"Max {compound}",
                        value=float(series.max()),
                        key=f"max_{compound}",
                    )
                    if min_val > max_val:
                        st.error("Minimum value cannot be greater than maximum value.")
                        continue
                    composition_filters[compound] = {
                        "mode": "range",
                        "min": float(min_val),
                        "max": float(max_val),
                    }

        if composition_filters:
            filtered_comp = apply_composition_filters(comp_df, composition_filters)
        st.caption(f"Matched composition rows: {len(filtered_comp)}")

    else:
        st.subheader("NER Tag Search")
        selected_ner_cols = st.multiselect(
            "NER tag columns to search",
            options=[col for col in NER_TAG_COLUMNS if col in ner_df.columns],
            default=[col for col in NER_TAG_COLUMNS if col in ner_df.columns],
        )
        keyword = st.text_input("Search text for NER tags", placeholder="e.g., chloride, slag, porosity")
        if selected_ner_cols:
            filtered_ner = apply_ner_filter(ner_df, selected_ner_cols, keyword)
        else:
            filtered_ner = ner_df.iloc[0:0].copy()
        st.caption(f"Matched papers (PII): {filtered_ner['PII'].nunique()}")

    if search_mode == "Composition":
        result_df = filtered_comp.merge(ner_df, left_on="pii", right_on="PII", how="left")
    else:
        result_df = comp_df.merge(filtered_ner, left_on="pii", right_on="PII", how="inner")

    result_df["abstract"] = result_df["pii"].map(abstract_map).fillna("")
    result_df["pii"] = result_df["pii"].fillna("").astype(str).map(
        lambda pii: f"{PII_BASE_URL}{pii}" if pii else ""
    )

    result_columns = (
        ["pii", "table_number"]
        + available_compounds
        + [col for col in ["mid", "total"] if col in result_df.columns]
        + [col for col in NER_TAG_COLUMNS if col in result_df.columns]
        + ["abstract"]
    )
    result_columns = [col for col in result_columns if col in result_df.columns]
    final_df = result_df[result_columns].copy()

    st.subheader(f"Results ({len(final_df)} rows)")
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
        label="Download results as CSV",
        data=final_df.to_csv(index=False).encode("utf-8"),
        file_name="composition_ner_abstract_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
