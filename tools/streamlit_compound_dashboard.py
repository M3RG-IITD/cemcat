import pandas as pd
import streamlit as st


st.set_page_config(page_title="Compound Filter Dashboard", layout="wide")

CSV_PATH = "upload/composition_cls_slag_updated.csv"
COMPOUND_COLUMNS = ["SiO2", "Al2O3", "Fe2O3", "CaO", "Na2O", "K2O", "MgO", "SO3"]
PII_BASE_URL = "https://www.sciencedirect.com/science/article/pii/"


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse identifiers from source fields for explicit output columns.
    if "mid" in df.columns:
        mid_raw = df["mid"].where(df["mid"].notna(), "")
        mid_parts = mid_raw.astype(str).str.split("_")
        df["pii"] = mid_parts.str[0]
        df["mid"] = mid_parts.str[-1]
    else:
        df["pii"] = ""
        df["mid"] = ""
    # User-defined rule: tid = 1 + last segment from `pii_tid` split by "_".
    tid_last_part = pd.to_numeric(df["pii_tid"].astype(str).str.split("_").str[-1], errors="coerce")
    df["table_number"] = (tid_last_part + 1).astype("Int64")

    for col in COMPOUND_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, spec in filters.items():
        if spec["mode"] == "exact":
            value = spec["value"]
            mask &= df[col].notna() & (df[col] == value)
        else:
            min_val = spec["min"]
            max_val = spec["max"]
            mask &= df[col].notna() & (df[col] >= min_val) & (df[col] <= max_val)
    return df[mask].copy()


def main() -> None:
    st.title("Chemical Compound Filter Dashboard")
    st.write(
        "Filter rows by exact value or value range for any compound column. "
        "Results include linked `pii` and `table_number`."
    )

    df = load_data(CSV_PATH)
    available_compounds = [col for col in COMPOUND_COLUMNS if col in df.columns]

    st.subheader("Filter Criteria")
    selected_filters: dict = {}

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

            series = df[compound].dropna()
            if series.empty:
                st.warning(f"No numeric values available for {compound}.")
                continue

            col_min = float(series.min())
            col_max = float(series.max())

            if mode == "Exact":
                exact_val = st.number_input(
                    f"Exact value ({compound})",
                    value=float(series.median()),
                    key=f"exact_{compound}",
                )
                selected_filters[compound] = {"mode": "exact", "value": float(exact_val)}
            else:
                c1, c2 = st.columns(2)
                min_val = c1.number_input(
                    f"Min {compound}",
                    value=col_min,
                    key=f"min_{compound}",
                )
                max_val = c2.number_input(
                    f"Max {compound}",
                    value=col_max,
                    key=f"max_{compound}",
                )
                if min_val > max_val:
                    st.error("Minimum value cannot be greater than maximum value.")
                    continue
                selected_filters[compound] = {
                    "mode": "range",
                    "min": float(min_val),
                    "max": float(max_val),
                }

    filtered_df = apply_filters(df, selected_filters) if selected_filters else df.copy()

    st.subheader("Applied Criteria")
    if selected_filters:
        st.json(selected_filters)
    else:
        st.info("No filters selected. Showing full dataset.")

    st.subheader(f"Results ({len(filtered_df)} rows)")

    output_columns = (
        ["pii", "table_number"]
        + available_compounds
        + [col for col in ["mid", "total"] if col in filtered_df.columns]
    )
    output_columns = [col for col in output_columns if col in filtered_df.columns]
    display_df = filtered_df[output_columns].copy()
    display_df["pii"] = display_df["pii"].fillna("").astype(str).map(
        lambda pii: f"{PII_BASE_URL}{pii}" if pii else ""
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "pii": st.column_config.LinkColumn(
                "pii",
                help="Open the article page on ScienceDirect",
                display_text=rf"{PII_BASE_URL}(.*)",
            )
        },
    )

    csv_download_df = filtered_df[output_columns].copy()
    csv_download_df["pii"] = csv_download_df["pii"].fillna("").astype(str).map(
        lambda pii: f"{PII_BASE_URL}{pii}" if pii else ""
    )
    csv_bytes = csv_download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered results as CSV",
        data=csv_bytes,
        file_name="filtered_composition_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
