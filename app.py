"""
Tableau de bord Streamlit - Sécurité Routière France (2024)
------------------------------------------------------
Projet : Data Integration & Applications (ST2DLDI)
Architecture : Medallion (Bronze -> Silver -> Gold)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sécurité Routière - Audit & Analyse",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Dictionnaires de référence (Mapping des catégories)
# ----------------------------------------------------------------------
LUM_LABELS = {1: "Plein jour", 2: "Crépuscule/Aube", 3: "Nuit (sans éclairage)", 4: "Nuit (éclairage éteint)", 5: "Nuit (éclairage allumé)"}
ATM_LABELS = {1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige/Grêle", 5: "Brouillard/Fumée", 6: "Vent fort", 7: "Temps éblouissant", 8: "Couvert", 9: "Autre"}
CATR_LABELS = {1: "Autoroute", 2: "Route Nationale", 3: "Route Départementale", 4: "Voie Communale", 5: "Hors réseau public", 6: "Parc de stationnement", 9: "Autre"}
CATU_LABELS = {1: "Conducteur", 2: "Passager", 3: "Piéton"}
GRAV_LABELS = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}

# ======================================================================
# COUCHE BRONZE : Chargement des données brutes
# ======================================================================
@st.cache_data(show_spinner="Extraction des données brutes (Couche Bronze)...")
def load_bronze_data():
    """Charge les données telles quelles depuis data.gouv.fr"""
    raw = {
        "caracteristiques": pd.read_csv("caract-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "lieux": pd.read_csv("lieux-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "usagers": pd.read_csv("usagers-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "vehicules": pd.read_csv("vehicules-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    }
    return raw

# ======================================================================
# DATA PROFILING (Analyse de la qualité sur la couche Bronze)
# ======================================================================
@st.cache_data(show_spinner="Exécution du Data Profiling...")
def compute_data_profiling(_raw):
    """Calcule toutes les métriques de qualité requises par le TP"""
    profiling = {}

    # 1. Structure et types
    profiling["structure"] = {name: {"lignes": df.shape[0], "colonnes": df.shape[1]} for name, df in _raw.items()}

    # 2. Valeurs manquantes (Vrais NaN + Sentinel -1)
    missing_data = {}
    for name, df in _raw.items():
        n = len(df)
        true_na = df.isna().sum()
        sentinel_na = (df.astype(str).apply(lambda x: x.str.strip() == "-1")).sum()

        df_missing = pd.DataFrame({
            "Valeurs Nulles (NaN)": true_na,
            "Non Renseigné (-1)": sentinel_na
        })
        df_missing["Total Manquant"] = df_missing["Valeurs Nulles (NaN)"] + df_missing["Non Renseigné (-1)"]
        df_missing["% Incomplet"] = (df_missing["Total Manquant"] / n * 100).round(2)
        missing_data[name] = df_missing[df_missing["Total Manquant"] > 0].sort_values("% Incomplet", ascending=False)

    profiling["missing"] = missing_data

    # 3. Anomalies de cohérence et validité (Coordinates & Ages)
    caract = _raw["caracteristiques"]
    usagers = _raw["usagers"]

    lat = pd.to_numeric(caract["lat"].astype(str).str.replace(",", "."), errors="coerce")
    lon = pd.to_numeric(caract["long"].astype(str).str.replace(",", "."), errors="coerce")
    profiling["coords_anomalies"] = int(((lat == 0) | (lon == 0) | lat.isna() | lon.isna()).sum())

    ages = 2024 - pd.to_numeric(usagers["an_nais"], errors="coerce")
    profiling["age_anomalies"] = int(((ages < 0) | (ages > 120)).sum())

    # 4. Doublons et Intégrité
    lieux = _raw["lieux"]
    acc_counts = lieux["Num_Acc"].value_counts()
    profiling["lieux_duplicates"] = int((acc_counts > 1).sum())

    ref_ids = set(caract["Num_Acc"])
    profiling["orphans"] = {
        "lieux": len(set(lieux["Num_Acc"]) - ref_ids),
        "usagers": len(set(usagers["Num_Acc"]) - ref_ids),
        "vehicules": len(set(_raw["vehicules"]["Num_Acc"]) - ref_ids)
    }

    return profiling

# ======================================================================
# COUCHES SILVER & GOLD : Transformations et Modélisation
# ======================================================================
@st.cache_data(show_spinner="Application de l'architecture Medallion (Silver -> Gold)...")
def process_medallion_pipeline(_raw):
    caract, lieux, usagers, vehicules = _raw["caracteristiques"].copy(), _raw["lieux"].copy(), _raw["usagers"].copy(), _raw["vehicules"].copy()

    # --- COUCHE SILVER : Nettoyage et Standardisation ---
    # 1. Standardisation des dates et coordonnées
    caract["date"] = pd.to_datetime(caract["an"].astype(str) + "-" + caract["mois"].astype(str) + "-" + caract["jour"].astype(str), errors="coerce")
    caract["datetime"] = pd.to_datetime(caract["date"].dt.strftime("%Y-%m-%d") + " " + caract["hrmn"], errors="coerce")
    caract["latitude"] = pd.to_numeric(caract["lat"].astype(str).str.replace(",", "."), errors="coerce")
    caract["longitude"] = pd.to_numeric(caract["long"].astype(str).str.replace(",", "."), errors="coerce")

    # 2. Remplacement des sentinelles (-1) par de vrais NaN
    for df in [lieux, usagers, vehicules]:
        for col in df.select_dtypes(include=np.number).columns:
            if col not in ["Num_Acc", "id_usager", "id_vehicule"]:
                df[col] = df[col].replace(-1, np.nan)

    # 3. Déduplication structurelle
    lieux = lieux.drop_duplicates(subset=["Num_Acc"], keep="first")

    # 4. Enrichissement
    caract["time_of_day"] = pd.cut(caract["datetime"].dt.hour, bins=[0, 6, 12, 18, 24], labels=["Nuit", "Matin", "Après-midi", "Soirée"], right=False)
    usagers["age"] = 2024 - usagers["an_nais"]

    worst_injury = usagers.groupby("Num_Acc")["grav"].max().rename("severity_index")
    caract = caract.merge(worst_injury, on="Num_Acc", how="left")

    # --- COUCHE GOLD : Modèle Analytique (Fait + Dimensions) ---
    fact = caract[["Num_Acc", "datetime", "time_of_day", "latitude", "longitude", "lum", "atm", "severity_index"]].copy()
    fact = fact.merge(lieux[["Num_Acc", "catr", "vma"]], on="Num_Acc", how="left")

    nb_veh = vehicules.groupby("Num_Acc").size().rename("nb_vehicules")
    nb_usg = usagers.groupby("Num_Acc").size().rename("nb_personnes")
    fact = fact.merge(nb_veh, on="Num_Acc", how="left").merge(nb_usg, on="Num_Acc", how="left")

    # Application des labels
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
# INTERFACE UTILISATEUR
# ======================================================================
st.title("🚦 Analyse Sécurité Routière (Pipeline Medallion)")
st.markdown("**Projet EFREI :** Data Profiling, Pipeline de Transformation & Visualisation (Bases de données 2024)")

try:
    raw_data = load_bronze_data()
    profiling_data = compute_data_profiling(raw_data)
    fact_acc, dim_veh, dim_usg = process_medallion_pipeline(raw_data)
except FileNotFoundError:
    st.error("⚠️ Fichiers CSV introuvables. Assurez-vous que caract-2024.csv, lieux-2024.csv, usagers-2024.csv et vehicules-2024.csv sont dans le même dossier.")
    st.stop()

tab_audit, tab_silver, tab_dashboard = st.tabs([
    "📊 1. Data Profiling (Couche Bronze)",
    "🛠️ 2. Transformations (Couche Silver)",
    "📈 3. Dashboard Analytique (Couche Gold)"
])

# ----------------------------------------------------------------------
# ONGLET 1 : DATA PROFILING
# ----------------------------------------------------------------------
with tab_audit:
    st.header("Partie 1 : Évaluation de la qualité des données brutes")

    st.subheader("A. Structure du Dataset")
    cols = st.columns(4)
    for i, (name, stats) in enumerate(profiling_data["structure"].items()):
        cols[i].metric(label=f"Table : {name}", value=f"{stats['lignes']:,} lignes", delta=f"{stats['colonnes']} colonnes", delta_color="off")

    st.markdown("---")

    st.subheader("B. Valeurs Manquantes et Complétude")
    st.info("💡 **Analyse Sémantique :** Dans ce jeu de données, la non-réponse est encodée par la valeur sentinelle `-1`. Pour un audit précis, nous cumulons les `NaN` (cellules vides) et les `-1`.")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Table : Caractéristiques**")
        st.dataframe(profiling_data["missing"]["caracteristiques"].head(5), use_container_width=True)
    with t2:
        st.markdown("**Table : Lieux**")
        st.dataframe(profiling_data["missing"]["lieux"].head(5), use_container_width=True)

    st.markdown("---")

    st.subheader("C. Cohérence et Validité (Anomalies Métier)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies GPS", f"{profiling_data['coords_anomalies']:,}", "Lat/Lon = 0 ou invalide", delta_color="inverse")
    c2.metric("Âges Aberrants", f"{profiling_data['age_anomalies']:,}", "Âge < 0 ou > 120", delta_color="inverse")
    c3.metric("Doublons structurels (Lieux)", f"{profiling_data['lieux_duplicates']:,}", "Accidents avec > 1 ligne", delta_color="inverse")

    st.markdown("---")

    st.subheader("D. Intégrité Référentielle")
    st.write("Vérification que chaque enregistrement enfant possède un identifiant `Num_Acc` valide dans la table mère (`caracteristiques`).")
    ref_df = pd.DataFrame(list(profiling_data["orphans"].items()), columns=["Table", "Lignes Orphelines"])
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    if ref_df["Lignes Orphelines"].sum() == 0:
        st.success("✅ Intégrité référentielle parfaite : 0 ligne orpheline.")

# ----------------------------------------------------------------------
# ONGLET 2 : STRATÉGIE DE REMÉDIATION (SILVER)
# ----------------------------------------------------------------------
with tab_silver:
    st.header("Partie 2 : Architecture Medallion & Remédiation")
    st.markdown("""
    Suite à l'audit, voici les transformations appliquées pour passer en couche **Silver** puis **Gold** :
    
    * **Standardisation :** Conversion des formats de date (fusion `an`, `mois`, `jour`, `hrmn` en un type `datetime`). Remplacement des virgules par des points pour les coordonnées GPS.
    * **Nettoyage :** Remplacement global de la valeur sentinelle `-1` par un vrai type `NaN` compréhensible par les modèles.
    * **Déduplication :** Suppression des multiples entrées pour un même `Num_Acc` dans la table `lieux` (qui fausserait les jointures).
    * **Enrichissement (Gold) :** * Création d'un `severity_index` global par accident (gravité maximale rencontrée parmi les usagers).
        * Création d'une dimension temporelle (`time_of_day`).
        * Calcul de l'âge dynamique (`2024 - an_nais`).
    """)

    st.image("https://databricks.com/wp-content/uploads/2022/03/medallion-architecture-1.png", width=600, caption="Concept de l'architecture Medallion")

# ----------------------------------------------------------------------
# ONGLET 3 : DASHBOARD (GOLD)
# ----------------------------------------------------------------------
with tab_dashboard:
    st.sidebar.header("Filtres Analytiques (Gold Layer)")

    gravite_filter = st.sidebar.multiselect("Sévérité de l'accident", options=GRAV_LABELS.values(), default=list(GRAV_LABELS.values()))
    moment_filter = st.sidebar.multiselect("Moment de la journée", options=["Matin", "Après-midi", "Soirée", "Nuit"], default=["Matin", "Après-midi", "Soirée", "Nuit"])

    filtered_fact = fact_acc[
        (fact_acc["Sévérité"].isin(gravite_filter)) &
        (fact_acc["time_of_day"].isin(moment_filter))
    ]

    st.header("Tableau de bord de l'Accidentologie (2024)")
    st.caption(f"Données filtrées : {len(filtered_fact):,} accidents affichés.")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Accidents", f"{len(filtered_fact):,}")
    k2.metric("Personnes Impliquées", f"{filtered_fact['nb_personnes'].sum():,.0f}")
    k3.metric("Véhicules Impliqués", f"{filtered_fact['nb_vehicules'].sum():,.0f}")
    pct_mortal = (filtered_fact["severity_index"] == 2).mean() * 100 if len(filtered_fact) > 0 else 0
    k4.metric("Taux de Mortalité", f"{pct_mortal:.1f}%")

    st.markdown("---")

    # Visualisations
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Accidents par Type de Route")
        fig_route = px.bar(
            filtered_fact["Type_Route"].value_counts().reset_index(),
            x="Type_Route", y="count", color="Type_Route",
            labels={"count": "Nombre d'accidents", "Type_Route": "Réseau Routier"}
        )
        fig_route.update_layout(showlegend=False)
        st.plotly_chart(fig_route, use_container_width=True)

    with col_chart2:
        st.subheader("Sévérité moyenne selon la Météo")
        sev_meteo = filtered_fact.groupby("Météo")["severity_index"].mean().reset_index().sort_values("severity_index", ascending=False)
        fig_meteo = px.bar(
            sev_meteo, x="Météo", y="severity_index", color="severity_index",
            color_continuous_scale="Reds", labels={"severity_index": "Indice de Gravité (1 à 4)"}
        )
        st.plotly_chart(fig_meteo, use_container_width=True)

    # Carte Mapbox
    st.subheader("Répartition Géographique")
    map_df = filtered_fact.dropna(subset=["latitude", "longitude"])
    map_df = map_df[(map_df["latitude"].between(41, 51.5)) & (map_df["longitude"].between(-5.5, 10))] # Filtre Métropole pour affichage

    if not map_df.empty:
        fig_map = px.scatter_map(
            map_df.sample(min(3000, len(map_df)), random_state=42),
            lat="latitude", lon="longitude", color="Sévérité",
            zoom=4.5, height=500, hover_data=["Type_Route", "time_of_day"]
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Aucune donnée géographique valide à afficher pour ces filtres.")