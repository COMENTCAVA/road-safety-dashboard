"""
Tableau de bord Streamlit - Sécurité Routière France (2024)
------------------------------------------------------
Architecture Medallion : Audit (Bronze) & Analytique (Gold)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sécurité Routière - EDA & Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ----------------------------------------------------------------------
# Dictionnaires de référence complets
# ----------------------------------------------------------------------
LUM_LABELS = {1: "Plein jour", 2: "Crépuscule/Aube", 3: "Nuit (sans éclairage)", 4: "Nuit (éclairage éteint)", 5: "Nuit (éclairage allumé)"}
ATM_LABELS = {1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige/Grêle", 5: "Brouillard/Fumée", 6: "Vent fort", 7: "Temps éblouissant", 8: "Couvert", 9: "Autre"}
CATR_LABELS = {1: "Autoroute", 2: "Route Nationale", 3: "Route Départementale", 4: "Voie Communale", 5: "Hors réseau public", 6: "Parc de stationnement", 9: "Autre"}
CATU_LABELS = {1: "Conducteur", 2: "Passager", 3: "Piéton"}
GRAV_LABELS = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
CATV_LABELS = {1: "Vélo", 2: "Cyclomoteur", 7: "Voiture", 10: "Véhicule utilitaire", 13: "Poids lourd (<7.5t)", 14: "Poids lourd (>=7.5t)", 30: "Scooter", 33: "Moto", 37: "Bus", 38: "Autocar"}

# ======================================================================
# COUCHE BRONZE : Chargement
# ======================================================================
@st.cache_data(show_spinner="Chargement des données brutes...")
def load_raw_data():
    return {
        "caracteristiques": pd.read_csv("caract-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "lieux": pd.read_csv("lieux-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "usagers": pd.read_csv("usagers-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "vehicules": pd.read_csv("vehicules-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    }

# ======================================================================
# PROFILING & QUALITÉ DES DONNÉES (Inspiré du Notebook)
# ======================================================================
@st.cache_data(show_spinner="Analyse de la qualité en profondeur...")
def deep_data_profiling(_raw):
    profiling = {}

    # 1. Volumétrie et Cardinalité
    profiling["volumetrie"] = {k: {"lignes": v.shape[0], "cols": v.shape[1], "acc_uniques": v["Num_Acc"].nunique()} for k, v in _raw.items()}

    # 2. Analyse de la complétude (Vrais NaN + Sentinelles -1 + Vides)
    missing_stats = {}
    for name, df in _raw.items():
        n = len(df)
        df_na = pd.DataFrame(index=df.columns)
        df_na["NaN_Classiques"] = df.isna().sum()
        df_na["Sentinelles_(-1)"] = (df.astype(str).apply(lambda x: x.str.strip() == "-1")).sum()
        df_na["Total_Manquant"] = df_na["NaN_Classiques"] + df_na["Sentinelles_(-1)"]
        df_na["%_Incomplet"] = (df_na["Total_Manquant"] / n * 100).round(2)
        missing_stats[name] = df_na[df_na["Total_Manquant"] > 0].sort_values("%_Incomplet", ascending=False)
    profiling["missing"] = missing_stats

    # 3. Valeurs Aberrantes (Outliers)
    caract = _raw["caracteristiques"]
    usagers = _raw["usagers"]

    # Géolocalisation
    lat = pd.to_numeric(caract["lat"].astype(str).str.replace(",", "."), errors="coerce")
    lon = pd.to_numeric(caract["long"].astype(str).str.replace(",", "."), errors="coerce")

    profiling["outliers"] = {
        "coords_nulles": int(((lat == 0) & (lon == 0)).sum()),
        "coords_hors_limites": int(((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)).sum()),
        "coords_manquantes": int(lat.isna().sum())
    }

    # Âges
    ages = 2024 - pd.to_numeric(usagers["an_nais"], errors="coerce")
    profiling["outliers"]["ages_negatifs"] = int((ages < 0).sum())
    profiling["outliers"]["ages_centenaires"] = int((ages > 105).sum())

    # 4. Intégrité Structurelle (Le problème de la table lieux)
    lieux = _raw["lieux"]
    acc_counts = lieux["Num_Acc"].value_counts()
    profiling["lieux_duplicates"] = {
        "accidents_multiples": int((acc_counts > 1).sum()),
        "max_lignes_pour_un_acc": int(acc_counts.max())
    }

    # 5. Intégrité Référentielle (Orphelins)
    ref_ids = set(caract["Num_Acc"])
    profiling["orphans"] = {
        name: len(set(_raw[name]["Num_Acc"]) - ref_ids) for name in ["lieux", "usagers", "vehicules"]
    }

    return profiling

# ======================================================================
# PIPELINE SILVER & GOLD
# ======================================================================
@st.cache_data(show_spinner="Construction du modèle analytique...")
def process_pipeline(_raw):
    caract, lieux, usagers, vehicules = _raw["caracteristiques"].copy(), _raw["lieux"].copy(), _raw["usagers"].copy(), _raw["vehicules"].copy()

    # --- NETTOYAGE (SILVER) ---
    caract["date"] = pd.to_datetime(caract["an"].astype(str) + "-" + caract["mois"].astype(str) + "-" + caract["jour"].astype(str), errors="coerce")
    caract["datetime"] = pd.to_datetime(caract["date"].dt.strftime("%Y-%m-%d") + " " + caract["hrmn"], errors="coerce")
    caract["latitude"] = pd.to_numeric(caract["lat"].astype(str).str.replace(",", "."), errors="coerce")
    caract["longitude"] = pd.to_numeric(caract["long"].astype(str).str.replace(",", "."), errors="coerce")

    for df in [lieux, usagers, vehicules]:
        for col in df.select_dtypes(include=np.number).columns:
            if col not in ["Num_Acc", "id_usager", "id_vehicule"]:
                df[col] = df[col].replace(-1, np.nan)

    lieux = lieux.drop_duplicates(subset=["Num_Acc"], keep="first")

    caract["time_of_day"] = pd.cut(caract["datetime"].dt.hour, bins=[0, 6, 12, 18, 24], labels=["Nuit", "Matin", "Après-midi", "Soirée"], right=False)
    usagers["age"] = 2024 - usagers["an_nais"]
    worst_injury = usagers.groupby("Num_Acc")["grav"].max().rename("severity_index")
    caract = caract.merge(worst_injury, on="Num_Acc", how="left")

    # --- MODÈLE (GOLD) ---
    fact = caract[["Num_Acc", "datetime", "time_of_day", "latitude", "longitude", "lum", "atm", "severity_index"]].copy()
    fact = fact.merge(lieux[["Num_Acc", "catr", "vma"]], on="Num_Acc", how="left")

    nb_veh = vehicules.groupby("Num_Acc").size().rename("nb_vehicules")
    nb_usg = usagers.groupby("Num_Acc").size().rename("nb_personnes")
    fact = fact.merge(nb_veh, on="Num_Acc", how="left").merge(nb_usg, on="Num_Acc", how="left")

    fact["Luminosité"] = fact["lum"].map(LUM_LABELS).fillna("Inconnu")
    fact["Météo"] = fact["atm"].map(ATM_LABELS).fillna("Inconnu")
    fact["Type_Route"] = fact["catr"].map(CATR_LABELS).fillna("Inconnu")
    fact["Sévérité"] = fact["severity_index"].map(GRAV_LABELS).fillna("Inconnu")

    dim_vehicule = vehicules[["id_vehicule", "Num_Acc", "catv"]].copy()
    dim_vehicule["Categorie"] = dim_vehicule["catv"].map(CATV_LABELS).fillna("Autre")

    dim_usager = usagers[["id_usager", "Num_Acc", "catu", "grav", "age", "sexe"]].copy()
    dim_usager["Role"] = dim_usager["catu"].map(CATU_LABELS).fillna("Inconnu")

    return fact, dim_vehicule, dim_usager

# ======================================================================
# INTERFACE PRINCIPALE
# ======================================================================
st.title("🚦 Analyse Exploratoire & Accidentologie (2024)")

try:
    raw_data = load_raw_data()
    profiling = deep_data_profiling(raw_data)
    fact_acc, dim_veh, dim_usg = process_pipeline(raw_data)
except FileNotFoundError:
    st.error("⚠️ Fichiers CSV introuvables. Vérifiez la présence des 4 fichiers à la racine.")
    st.stop()

tab_dq, tab_dash = st.tabs(["🔬 Exploratory Data Analysis (Data Quality)", "📊 Dashboard Analytique"])

# ----------------------------------------------------------------------
# ONGLET 1 : DATA QUALITY (EDA)
# ----------------------------------------------------------------------
with tab_dq:
    st.header("Analyse Exploratoire et Qualité des Données (EDA)")
    st.markdown("Cette section présente l'audit approfondi du jeu de données brut avant toute transformation, mettant en évidence les biais potentiels pour la modélisation.")

    # 1. Volumétrie
    st.subheader("1. Vue d'ensemble & Cardinalité")
    vol_df = pd.DataFrame.from_dict(profiling["volumetrie"], orient="index")
    vol_df.columns = ["Lignes", "Colonnes", "Accidents Uniques (Num_Acc)"]
    st.dataframe(vol_df, use_container_width=True)

    st.markdown("---")

    # 2. Complétude
    st.subheader("2. Analyse de la Complétude (Missing Values)")
    st.markdown("La gestion des valeurs manquantes est critique. Ici, les non-réponses sont souvent encodées par la valeur sentinelle `-1`. Le graphique ci-dessous fusionne les `NaN` natifs et les `-1` pour révéler le véritable taux de données inexploitables par table.")

    all_missing = []
    for table_name, df_miss in profiling["missing"].items():
        for col, row in df_miss.iterrows():
            if row["%_Incomplet"] > 5:  # On n'affiche que les colonnes > 5% manquants pour la lisibilité
                all_missing.append({"Table": table_name, "Colonne": col, "% Incomplet": row["%_Incomplet"]})

    if all_missing:
        miss_df = pd.DataFrame(all_missing).sort_values("% Incomplet", ascending=False)
        fig_miss = px.bar(
            miss_df, x="% Incomplet", y="Colonne", color="Table", orientation="h",
            title="Colonnes présentant plus de 5% de valeurs manquantes ou sentinelles (-1)",
            height=400
        )
        fig_miss.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_miss, use_container_width=True)

    st.markdown("---")

    # 3. Outliers & Validité
    st.subheader("3. Validité Métier & Valeurs Aberrantes (Outliers)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Qualité de la Géolocalisation**")
        st.write(f"- Coordonnées absentes (NaN) : **{profiling['outliers']['coords_manquantes']:,}**")
        st.write(f"- Coordonnées à (0, 0) : **{profiling['outliers']['coords_nulles']:,}**")
        st.write(f"- Coordonnées hors limites (Lat/Lon) : **{profiling['outliers']['coords_hors_limites']:,}**")
        st.info("💡 Les modèles spatiaux nécessiteront un filtrage strict sur les bounding boxes de la France métropolitaine et des DROM-COM.")

    with c2:
        st.markdown("**Validité Démographique (Âges)**")
        st.write(f"- Âges négatifs calculés : **{profiling['outliers']['ages_negatifs']:,}**")
        st.write(f"- Usagers centenaires (>105 ans) : **{profiling['outliers']['ages_centenaires']:,}**")
        st.warning("⚠️ Les anomalies démographiques devront être imputées ou exclues lors de la création de features (Feature Engineering).")

    st.markdown("---")

    # 4. Intégrité
    st.subheader("4. Intégrité Structurelle & Référentielle")
    i1, i2 = st.columns(2)

    with i1:
        st.markdown("**Anomalie de la table `lieux`**")
        st.error(f"Structure de données défectueuse détectée : **{profiling['lieux_duplicates']['accidents_multiples']:,}** accidents possèdent plusieurs lignes de description (jusqu'à {profiling['lieux_duplicates']['max_lignes_pour_un_acc']} lignes pour un seul `Num_Acc`). **Action requise :** Déduplication stricte avant jointure pour éviter l'explosion combinatoire.")

    with i2:
        st.markdown("**Clés étrangères (Orphelins)**")
        ref_df = pd.DataFrame.from_dict(profiling["orphans"], orient="index", columns=["Lignes sans correspondance parent"])
        st.dataframe(ref_df, use_container_width=True)
        if ref_df["Lignes sans correspondance parent"].sum() == 0:
            st.success("✅ Intégrité relationnelle validée : toutes les clés enfants pointent vers un accident existant.")

# ----------------------------------------------------------------------
# ONGLET 2 : DASHBOARD
# ----------------------------------------------------------------------
with tab_dash:
    st.sidebar.header("Filtres Analytiques")

    grav_options = sorted(fact_acc["Sévérité"].dropna().unique())
    gravite_filter = st.sidebar.multiselect("Sévérité de l'accident", options=grav_options, default=grav_options)

    time_options = ["Matin", "Après-midi", "Soirée", "Nuit"]
    moment_filter = st.sidebar.multiselect("Moment de la journée", options=time_options, default=time_options)

    filtered_fact = fact_acc[
        (fact_acc["Sévérité"].isin(gravite_filter)) &
        (fact_acc["time_of_day"].isin(moment_filter))
    ]

    st.header("Dashboard Analytique des Accidents")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accidents filtrés", f"{len(filtered_fact):,}")
    k2.metric("Personnes Impliquées", f"{filtered_fact['nb_personnes'].sum():,.0f}")
    k3.metric("Véhicules Impliqués", f"{filtered_fact['nb_vehicules'].sum():,.0f}")
    pct_mortal = (filtered_fact["severity_index"] == 2).mean() * 100 if len(filtered_fact) > 0 else 0
    k4.metric("Taux de Mortalité", f"{pct_mortal:.1f}%")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Typologie du Réseau Routier")
        fig_route = px.bar(
            filtered_fact["Type_Route"].value_counts().reset_index(),
            x="Type_Route", y="count", color="Type_Route",
            labels={"count": "Volume", "Type_Route": "Réseau"}
        )
        fig_route.update_layout(showlegend=False)
        st.plotly_chart(fig_route, use_container_width=True)

    with col_chart2:
        st.subheader("Impact des Conditions Météorologiques")
        sev_meteo = filtered_fact.groupby("Météo")["severity_index"].mean().reset_index().sort_values("severity_index", ascending=False)
        fig_meteo = px.bar(
            sev_meteo, x="Météo", y="severity_index", color="severity_index",
            color_continuous_scale="Reds", labels={"severity_index": "Indice de Gravité Moyen"}
        )
        st.plotly_chart(fig_meteo, use_container_width=True)

    # Carte Mapbox (Filtrée sur la métropole pour lisibilité)
    st.subheader("Cartographie des Sinistres")
    map_df = filtered_fact.dropna(subset=["latitude", "longitude"])
    map_df = map_df[(map_df["latitude"].between(41, 51.5)) & (map_df["longitude"].between(-5.5, 10))]

    if not map_df.empty:
        fig_map = px.scatter_map(
            map_df.sample(min(4000, len(map_df)), random_state=42),
            lat="latitude", lon="longitude", color="Sévérité",
            zoom=4.5, height=500, hover_data=["Type_Route", "time_of_day"]
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Aucune donnée géographique exploitable pour cette sélection.")