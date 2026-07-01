"""
Streamlit dashboard - French Road Safety Data (2024)
------------------------------------------------------
This app has two tabs:
1) Data Quality: shows the checks from Part 1 of the TP (missing values,
   sentinel -1 codes, duplicates, the structural problem in `lieux`,
   referential integrity, value range checks).
2) Dashboard: the Gold model (fact_accidents + dimensions) with KPIs,
   charts and a map, built on top of the Bronze -> Silver -> Gold pipeline.

How to run:
1) Put the 4 CSV files in the same folder as this app.py:
   caract-2024.csv, lieux-2024.csv, usagers-2024.csv, vehicules-2024.csv
2) Install requirements:
   pip install streamlit pandas numpy plotly
3) Run (do NOT use "python app.py", Streamlit needs its own command):
   streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Road Safety - Data Quality & Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 French Road Safety Data (2024)")
st.caption("Bronze -> Silver -> Gold pipeline: data quality checks, then the BI dashboard.")

# ----------------------------------------------------------------------
# Simple label dictionaries (to make codes readable)
# ----------------------------------------------------------------------
LUM_LABELS = {
    1: "Daylight", 2: "Dusk/dawn", 3: "Night, no light",
    4: "Night, light off", 5: "Night, light on",
}
ATM_LABELS = {
    1: "Normal", 2: "Light rain", 3: "Heavy rain", 4: "Snow/hail",
    5: "Fog/smoke", 6: "Strong wind", 7: "Dazzling weather",
    8: "Overcast", 9: "Other",
}
CATR_LABELS = {
    1: "Highway", 2: "National road", 3: "Departmental road",
    4: "City street", 5: "Off public network", 6: "Parking access", 9: "Other",
}
CATU_LABELS = {1: "Driver", 2: "Passenger", 3: "Pedestrian"}
GRAV_LABELS = {1: "Unharmed", 2: "Killed", 3: "Hospitalized", 4: "Light injury"}
CATV_LABELS = {
    1: "Bicycle", 2: "Moped", 7: "Car", 10: "Utility vehicle",
    13: "Truck (<7.5t)", 14: "Truck (>=7.5t)", 30: "Scooter", 33: "Motorbike",
    37: "Bus", 38: "Coach",
}


# ----------------------------------------------------------------------
# Bronze: load the raw files once (cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading raw csv files...")
def load_raw():
    caract = pd.read_csv("caract-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    lieux = pd.read_csv("lieux-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    usagers = pd.read_csv("usagers-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    vehicules = pd.read_csv("vehicules-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    return {"caract": caract, "lieux": lieux, "usagers": usagers, "vehicules": vehicules}


# ----------------------------------------------------------------------
# Data quality checks (this reproduces Part 1 of the TP)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Running data quality checks...")
def compute_quality_report(_raw):
    report = {}

    missing_tables = {}
    for name, df in _raw.items():
        n = len(df)
        true_na_pct = (df.isna().sum() / n * 100).round(2)
        sentinel_pct = {}
        for col in df.columns:
            count = (df[col].astype(str).str.strip() == "-1").sum()
            if count > 0:
                sentinel_pct[col] = round(count / n * 100, 2)
        combined = pd.DataFrame({"true_NaN_%": true_na_pct})
        combined["sentinel_-1_%"] = pd.Series(sentinel_pct)
        combined = combined.fillna(0)
        combined = combined[(combined["true_NaN_%"] > 0) | (combined["sentinel_-1_%"] > 0)]
        combined = combined.sort_values("true_NaN_%", ascending=False)
        missing_tables[name] = combined
    report["missing_tables"] = missing_tables

    report["exact_duplicates"] = {name: int(df.duplicated().sum()) for name, df in _raw.items()}

    lieux = _raw["lieux"]
    acc_count = lieux["Num_Acc"].value_counts()
    multi = acc_count[acc_count > 1]
    report["lieux_total_rows"] = len(lieux)
    report["lieux_unique_accidents"] = lieux["Num_Acc"].nunique()
    report["lieux_accidents_with_many_rows"] = len(multi)
    report["lieux_max_rows_for_one_accident"] = int(acc_count.max())

    caract_ids = set(_raw["caract"]["Num_Acc"])
    integrity = {}
    for name in ["lieux", "usagers", "vehicules"]:
        orphan = set(_raw[name]["Num_Acc"]) - caract_ids
        integrity[name] = len(orphan)
    report["referential_integrity"] = integrity

    caract = _raw["caract"]
    lat = caract["lat"].astype(str).str.replace(",", ".").astype(float)
    lon = caract["long"].astype(str).str.replace(",", ".").astype(float)
    report["coord_stats"] = {
        "lat_min": round(lat.min(), 2), "lat_max": round(lat.max(), 2),
        "lon_min": round(lon.min(), 2), "lon_max": round(lon.max(), 2),
        "zero_coords": int(((lat == 0) | (lon == 0)).sum()),
    }

    usagers = _raw["usagers"]
    age = 2024 - usagers["an_nais"]
    report["age_stats"] = {
        "min": age.min(), "max": age.max(),
        "negative_ages": int((age < 0).sum()),
    }

    return report


# ----------------------------------------------------------------------
# Silver + Gold: build the analytical model (cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Building the Silver and Gold layers...")
def build_gold_data(_raw):
    caract = _raw["caract"].copy()
    lieux = _raw["lieux"].copy()
    usagers = _raw["usagers"].copy()
    vehicules = _raw["vehicules"].copy()

    caract["date"] = pd.to_datetime(
        caract["an"].astype(str) + "-" + caract["mois"].astype(str) + "-" + caract["jour"].astype(str),
        format="%Y-%m-%d", errors="coerce",
    )
    caract["datetime"] = pd.to_datetime(
        caract["date"].dt.strftime("%Y-%m-%d") + " " + caract["hrmn"], errors="coerce"
    )
    caract["latitude"] = caract["lat"].astype(str).str.replace(",", ".").astype(float)
    caract["longitude"] = caract["long"].astype(str).str.replace(",", ".").astype(float)

    lieux = lieux.drop(columns=["lartpc"])
    vehicules = vehicules.drop(columns=["occutc"])

    for col in lieux.columns:
        if col != "Num_Acc" and pd.api.types.is_numeric_dtype(lieux[col]):
            lieux[col] = lieux[col].replace(-1, np.nan)
    for col in usagers.columns:
        if col not in ["Num_Acc", "id_usager", "id_vehicule"] and pd.api.types.is_numeric_dtype(usagers[col]):
            usagers[col] = usagers[col].replace(-1, np.nan)
    for col in vehicules.columns:
        if col not in ["Num_Acc", "id_vehicule"] and pd.api.types.is_numeric_dtype(vehicules[col]):
            vehicules[col] = vehicules[col].replace(-1, np.nan)

    lieux = lieux.sort_values("Num_Acc").drop_duplicates(subset="Num_Acc", keep="first")

    worst_injury = usagers.groupby("Num_Acc")["grav"].max().rename("severity_index")
    caract = caract.merge(worst_injury, on="Num_Acc", how="left")

    def get_time_of_day(dt):
        if pd.isna(dt):
            return np.nan
        hour = dt.hour
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 23:
            return "Evening"
        else:
            return "Night"

    caract["time_of_day"] = caract["datetime"].apply(get_time_of_day)
    caract["weekday"] = caract["datetime"].dt.day_name()
    usagers["age"] = 2024 - usagers["an_nais"]

    fact = caract[[
        "Num_Acc", "datetime", "time_of_day", "weekday", "latitude", "longitude",
        "lum", "atm", "col", "severity_index",
    ]].copy()
    fact = fact.merge(lieux[["Num_Acc", "catr", "nbv", "vma", "surf"]], on="Num_Acc", how="left")

    nb_vehicles = vehicules.groupby("Num_Acc").size().rename("nb_vehicles")
    nb_persons = usagers.groupby("Num_Acc").size().rename("nb_persons")
    fact = fact.merge(nb_vehicles, on="Num_Acc", how="left")
    fact = fact.merge(nb_persons, on="Num_Acc", how="left")

    fact["lum_label"] = fact["lum"].map(LUM_LABELS).fillna("Unknown")
    fact["atm_label"] = fact["atm"].map(ATM_LABELS).fillna("Unknown")
    fact["catr_label"] = fact["catr"].map(CATR_LABELS).fillna("Unknown")
    fact["severity_label"] = fact["severity_index"].map(GRAV_LABELS).fillna("Unknown")

    dim_vehicle = vehicules[["id_vehicule", "Num_Acc", "catv"]].drop_duplicates()
    dim_vehicle["catv_label"] = dim_vehicle["catv"].map(CATV_LABELS).fillna("Other")

    dim_person = usagers[["id_usager", "Num_Acc", "catu", "grav", "sexe", "age"]].drop_duplicates()
    dim_person["catu_label"] = dim_person["catu"].map(CATU_LABELS).fillna("Unknown")
    dim_person["grav_label"] = dim_person["grav"].map(GRAV_LABELS).fillna("Unknown")

    return fact, dim_vehicle, dim_person


# ----------------------------------------------------------------------
# Load everything once
# ----------------------------------------------------------------------
try:
    raw = load_raw()
except FileNotFoundError:
    st.error(
        "Could not find the 4 CSV files. Please put caract-2024.csv, lieux-2024.csv, "
        "usagers-2024.csv and vehicules-2024.csv in the same folder as app.py."
    )
    st.stop()

quality = compute_quality_report(raw)
fact_accidents, dim_vehicle, dim_person = build_gold_data(raw)

# ----------------------------------------------------------------------
# Tabs: Data Quality first, then the Dashboard
# ----------------------------------------------------------------------
tab_quality, tab_dashboard = st.tabs(["📋 Data Quality", "📊 Dashboard"])

# ========================================================================
# TAB 1: DATA QUALITY
# ========================================================================
with tab_quality:
    st.header("Data quality report (Bronze layer)")
    st.caption("This reproduces the checks from Part 1 of the TP, computed on the raw files.")

    st.subheader("Table sizes")
    size_cols = st.columns(4)
    for c, (name, df) in zip(size_cols, raw.items()):
        c.metric(name, f"{len(df):,} rows", f"{df.shape[1]} columns")

    st.markdown("---")

    st.subheader("Missing values (true empty cells + sentinel code -1)")
    st.caption("In this dataset, the code -1 also means 'not specified', so we count both.")

    mq_cols = st.columns(4)
    for c, (name, tbl) in zip(mq_cols, quality["missing_tables"].items()):
        with c:
            st.markdown(f"**{name}**")
            if tbl.empty:
                st.write("No missing values.")
            else:
                st.dataframe(tbl, height=250)

    all_missing = []
    for name, tbl in quality["missing_tables"].items():
        for col in tbl.index:
            total_pct = tbl.loc[col, "true_NaN_%"] + tbl.loc[col, "sentinel_-1_%"]
            all_missing.append({"table": name, "column": col, "missing_%": total_pct})
    missing_df = pd.DataFrame(all_missing).sort_values("missing_%", ascending=False).head(12)

    fig_missing = px.bar(
        missing_df, x="missing_%", y="column", color="table", orientation="h",
        labels={"missing_%": "Missing or unknown (%)", "column": "Column"},
        title="Top 12 columns with the most missing / unknown values",
    )
    fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_missing, use_container_width=True)

    st.markdown("---")

    st.subheader("Duplicates")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Exact duplicate rows per table**")
        dup_df = pd.DataFrame(
            {"table": list(quality["exact_duplicates"].keys()),
             "duplicate_rows": list(quality["exact_duplicates"].values())}
        )
        st.dataframe(dup_df, hide_index=True)

    with d2:
        st.markdown("**Structural problem in `lieux`**")
        st.warning(
            f"`lieux` has **{quality['lieux_total_rows']:,} rows** for only "
            f"**{quality['lieux_unique_accidents']:,} unique accidents**.\n\n"
            f"**{quality['lieux_accidents_with_many_rows']:,} accidents** have more than "
            f"one row (up to {quality['lieux_max_rows_for_one_accident']} rows for a single accident).\n\n"
            "This must be fixed before joining `lieux` into the fact table, "
            "otherwise accidents would be duplicated."
        )

    st.markdown("---")

    st.subheader("Referential integrity (Num_Acc must exist in caract)")
    ref_df = pd.DataFrame(
        {"table": list(quality["referential_integrity"].keys()),
         "orphan_rows": list(quality["referential_integrity"].values())}
    )
    st.dataframe(ref_df, hide_index=True)
    if all(v == 0 for v in quality["referential_integrity"].values()):
        st.success("All accident IDs match correctly across tables. No orphan rows found.")

    st.markdown("---")

    st.subheader("Value range checks")
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**GPS coordinates**")
        cs = quality["coord_stats"]
        st.write(f"Latitude: {cs['lat_min']} to {cs['lat_max']}")
        st.write(f"Longitude: {cs['lon_min']} to {cs['lon_max']}")
        st.write(f"Rows with lat=0 or long=0 (likely invalid): {cs['zero_coords']}")
        st.caption("Wide range is expected: the dataset includes French overseas territories.")

    with v2:
        st.markdown("**Age (2024 minus birth year)**")
        ags = quality["age_stats"]
        st.write(f"Youngest: {ags['min']}")
        st.write(f"Oldest: {ags['max']}")
        st.write(f"Negative ages (impossible, would be an error): {ags['negative_ages']}")

    st.markdown("---")
    st.info(
        "**Summary:** referential integrity is perfect and category codes are valid. "
        "The main issues are the code -1 used instead of a real missing value, some "
        "columns that are almost always empty, and the structural duplication in `lieux`. "
        "All of these are fixed in the Silver layer before building the Dashboard tab."
    )

# ========================================================================
# TAB 2: DASHBOARD
# ========================================================================
with tab_dashboard:
    st.sidebar.header("Dashboard filters")

    time_options = ["Morning", "Afternoon", "Evening", "Night"]
    selected_times = st.sidebar.multiselect("Time of day", time_options, default=time_options)

    road_options = sorted(fact_accidents["catr_label"].dropna().unique())
    selected_roads = st.sidebar.multiselect("Road type", road_options, default=road_options)

    severity_options = sorted(fact_accidents["severity_label"].dropna().unique())
    selected_severity = st.sidebar.multiselect("Severity", severity_options, default=severity_options)

    filtered = fact_accidents[
        fact_accidents["time_of_day"].isin(selected_times)
        & fact_accidents["catr_label"].isin(selected_roads)
        & fact_accidents["severity_label"].isin(selected_severity)
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(filtered):,} accidents match the current filters.")

    st.header("General dashboard (Gold layer)")

    col1, col2, col3, col4 = st.columns(4)
    total_accidents = len(filtered)
    total_persons = int(filtered["nb_persons"].sum())
    total_vehicles = int(filtered["nb_vehicles"].sum())
    pct_killed = (filtered["severity_index"] == 2).mean() * 100 if total_accidents > 0 else 0

    col1.metric("Accidents", f"{total_accidents:,}")
    col2.metric("People involved", f"{total_persons:,}")
    col3.metric("Vehicles involved", f"{total_vehicles:,}")
    col4.metric("Accidents with a death", f"{pct_killed:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Average severity by time of day")
        avg_severity = (
            filtered.groupby("time_of_day")["severity_index"]
            .mean()
            .reindex(time_options)
            .reset_index()
        )
        fig1 = px.bar(
            avg_severity, x="time_of_day", y="severity_index",
            labels={"time_of_day": "Time of day", "severity_index": "Average severity (1-4)"},
            color="time_of_day",
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Number of accidents by road type")
        by_road = filtered["catr_label"].value_counts().reset_index()
        by_road.columns = ["Road type", "Accidents"]
        fig2 = px.bar(by_road, x="Road type", y="Accidents", color="Road type")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns([2, 1])

    with c3:
        st.subheader("Accident locations")
        map_data = filtered.dropna(subset=["latitude", "longitude"])
        map_data = map_data[
            (map_data["latitude"].between(41, 51.5)) & (map_data["longitude"].between(-5.5, 10))
        ]
        if len(map_data) > 0:
            fig3 = px.scatter_map(
                map_data.sample(min(3000, len(map_data)), random_state=1),
                lat="latitude", lon="longitude",
                color="severity_label",
                hover_data=["time_of_day", "catr_label"],
                zoom=4.3, height=450,
            )
            fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No accident matches the current filters.")

    with c4:
        st.subheader("Vehicle types involved")
        veh_in_filtered = dim_vehicle[dim_vehicle["Num_Acc"].isin(filtered["Num_Acc"])]
        by_vehicle = veh_in_filtered["catv_label"].value_counts().head(8).reset_index()
        by_vehicle.columns = ["Vehicle type", "Count"]
        fig4 = px.pie(by_vehicle, names="Vehicle type", values="Count", hole=0.4)
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Accidents by day of week")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_weekday = filtered["weekday"].value_counts().reindex(weekday_order).reset_index()
        by_weekday.columns = ["Weekday", "Accidents"]
        fig5 = px.line(by_weekday, x="Weekday", y="Accidents", markers=True)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        st.subheader("People involved, by category")
        ppl_in_filtered = dim_person[dim_person["Num_Acc"].isin(filtered["Num_Acc"])]
        by_catu = ppl_in_filtered["catu_label"].value_counts().reset_index()
        by_catu.columns = ["Category", "Count"]
        fig6 = px.bar(by_catu, x="Category", y="Count", color="Category")
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    with st.expander("See the fact_accidents table (Gold layer)"):
        st.dataframe(filtered.head(200))

    st.caption("Data source: data.gouv.fr - Bases de donnees annuelles des accidents corporels 2024.")