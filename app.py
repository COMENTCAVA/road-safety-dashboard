"""
Tableau de Bord Analytique et Pipeline ETL - Accidentologie France (2024)
-------------------------------------------------------------------------
Architecture Medallion (Bronze -> Silver -> Gold)
Intégration d'une Analyse Exploratoire des Données (EDA) exhaustive
et préparation des features pour la modélisation prédictive.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ======================================================================
# CONFIGURATION DE L'APPLICATION
# ======================================================================
st.set_page_config(
    page_title="Data Science - Pipeline Sécurité Routière",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# DICTIONNAIRES DE RÉFÉRENCE (METADATA)
# ======================================================================
LUM_LABELS = {1: "Plein jour", 2: "Crépuscule/Aube", 3: "Nuit (sans éclairage)", 4: "Nuit (éclairage éteint)", 5: "Nuit (éclairage allumé)"}
ATM_LABELS = {1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige/Grêle", 5: "Brouillard/Fumée", 6: "Vent fort", 7: "Temps éblouissant", 8: "Couvert", 9: "Autre"}
CATR_LABELS = {1: "Autoroute", 2: "Route Nationale", 3: "Route Départementale", 4: "Voie Communale", 5: "Hors réseau", 6: "Parking", 9: "Autre"}
CATU_LABELS = {1: "Conducteur", 2: "Passager", 3: "Piéton"}
GRAV_LABELS = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
CATV_LABELS = {1: "Vélo", 2: "Cyclomoteur", 7: "Voiture", 10: "Utilitaire", 13: "PL (<7.5t)", 14: "PL (>=7.5t)", 30: "Scooter", 33: "Moto", 37: "Bus", 38: "Autocar"}
SEXE_LABELS = {1: "Masculin", 2: "Féminin", -1: "Non renseigné"}
JOURS_FR = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi", "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"}

# ======================================================================
# COUCHE BRONZE : EXTRACTION ET OPTIMISATION MÉMOIRE
# ======================================================================
@st.cache_data(show_spinner="Extraction Bronze et optimisation de la RAM...")
def load_and_optimize_raw_data():
    """Charge les données et optimise les types pour réduire le memory footprint"""
    start_time = time.time()

    # Lecture brute
    raw = {
        "caracteristiques": pd.read_csv("caract-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "lieux": pd.read_csv("lieux-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "usagers": pd.read_csv("usagers-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "vehicules": pd.read_csv("vehicules-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    }

    # Calcul de la taille en mémoire initiale
    mem_initial = sum(df.memory_usage(deep=True).sum() for df in raw.values()) / (1024**2)

    # Optimisation basique des types numériques pour simuler un pipeline Data Eng
    for name, df in raw.items():
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')

    mem_final = sum(df.memory_usage(deep=True).sum() for df in raw.values()) / (1024**2)
    exec_time = time.time() - start_time

    return raw, mem_initial, mem_final, exec_time

# ======================================================================
# COUCHE SILVER & GOLD : ETL ET FEATURE ENGINEERING
# ======================================================================
@st.cache_data(show_spinner="Exécution du Pipeline ETL (Silver -> Gold)...")
def execute_etl_pipeline(_raw):
    caract = _raw["caracteristiques"].copy()
    lieux = _raw["lieux"].copy()
    usagers = _raw["usagers"].copy()
    vehicules = _raw["vehicules"].copy()

    # ---------------------------------------------------------
    # SILVER : Nettoyage & Standardisation
    # ---------------------------------------------------------
    # Dates et géométrie
    caract["date"] = pd.to_datetime(caract["an"].astype(str) + "-" + caract["mois"].astype(str) + "-" + caract["jour"].astype(str), errors="coerce")
    caract["datetime"] = pd.to_datetime(caract["date"].dt.strftime("%Y-%m-%d") + " " + caract["hrmn"], errors="coerce")
    caract["latitude"] = pd.to_numeric(caract["lat"].astype(str).str.replace(",", "."), errors="coerce")
    caract["longitude"] = pd.to_numeric(caract["long"].astype(str).str.replace(",", "."), errors="coerce")

    # Traitement des Sentinelles (-1)
    for df in [lieux, usagers, vehicules]:
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col not in ["Num_Acc", "id_usager", "id_vehicule"]:
                df.loc[df[col] == -1, col] = np.nan

    # Déduplication stricte
    lieux = lieux.drop_duplicates(subset=["Num_Acc"], keep="first")

    # ---------------------------------------------------------
    # FEATURE ENGINEERING (Préparation ML)
    # ---------------------------------------------------------
    usagers["age"] = 2024 - pd.to_numeric(usagers["an_nais"], errors="coerce")
    caract["heure"] = caract["datetime"].dt.hour

    # Contournement robuste de l'erreur locale : Mapping manuel
    caract["jour_semaine_en"] = caract["datetime"].dt.day_name()
    caract["jour_semaine"] = caract["jour_semaine_en"].map(JOURS_FR)

    caract["time_of_day"] = pd.cut(caract["heure"], bins=[0, 6, 12, 18, 24], labels=["Nuit", "Matin", "Après-midi", "Soirée"], right=False)

    # Target Variable : Index de sévérité globale du sinistre
    worst_injury = usagers.groupby("Num_Acc")["grav"].max().rename("severity_index")
    caract = caract.merge(worst_injury, on="Num_Acc", how="left")

    # ---------------------------------------------------------
    # GOLD : Modélisation Dimensionnelle
    # ---------------------------------------------------------
    # Table de faits
    fact = caract[["Num_Acc", "datetime", "heure", "time_of_day", "jour_semaine", "latitude", "longitude", "lum", "atm", "severity_index"]].copy()
    fact = fact.merge(lieux[["Num_Acc", "catr", "vma", "surf", "prof"]], on="Num_Acc", how="left")

    # Agrégations
    nb_veh = vehicules.groupby("Num_Acc").size().rename("nb_vehicules")
    nb_usg = usagers.groupby("Num_Acc").size().rename("nb_personnes")
    fact = fact.merge(nb_veh, on="Num_Acc", how="left").merge(nb_usg, on="Num_Acc", how="left")

    # Mapping métier pour BI
    fact["Luminosité"] = fact["lum"].map(LUM_LABELS).fillna("Inconnu")
    fact["Météo"] = fact["atm"].map(ATM_LABELS).fillna("Inconnu")
    fact["Type_Route"] = fact["catr"].map(CATR_LABELS).fillna("Inconnu")
    fact["Sévérité"] = fact["severity_index"].map(GRAV_LABELS).fillna("Inconnu")

    # Tables de dimensions
    dim_veh = vehicules[["id_vehicule", "Num_Acc", "catv", "obs", "choc", "motor"]].copy()
    dim_veh["Categorie"] = dim_veh["catv"].map(CATV_LABELS).fillna("Autre")

    dim_usg = usagers[["id_usager", "Num_Acc", "catu", "grav", "age", "sexe", "trajet", "secu1"]].copy()
    dim_usg["Role"] = dim_usg["catu"].map(CATU_LABELS).fillna("Inconnu")
    dim_usg["Sexe_Label"] = dim_usg["sexe"].map(SEXE_LABELS).fillna("Inconnu")

    return fact, dim_veh, dim_usg

# ======================================================================
# INITIALISATION ET CHARGEMENT
# ======================================================================
try:
    raw_data, mem_init, mem_fin, t_exec = load_and_optimize_raw_data()
    fact_acc, dim_veh, dim_usg = execute_etl_pipeline(raw_data)
except FileNotFoundError:
    st.error("⚠️ Fichiers CSV introuvables. Vérifiez le répertoire source.")
    st.stop()

# ======================================================================
# INTERFACE UTILISATEUR
# ======================================================================
st.title("🚦 Pipeline Data & Intelligence Artificielle - Accidentologie")
st.markdown("Une plateforme analytique bout-en-bout : de l'audit de qualité des données brutes jusqu'à l'exploration multidimensionnelle préparatoire au Machine Learning.")

tab_profiling, tab_univariate, tab_multivariate, tab_dashboard = st.tabs([
    "🛠️ Data Profiling & ETL",
    "📈 EDA : Univariée",
    "🔗 EDA : Bivariée & Corrélations",
    "📊 Business Intelligence"
])

# ----------------------------------------------------------------------
# ONGLET 1 : DATA PROFILING & PIPELINE ETL
# ----------------------------------------------------------------------
with tab_profiling:
    st.header("1. Métriques du Pipeline et Qualité de Données")

    st.subheader("1.1. Performances d'Extraction (Data Engineering)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Temps d'exécution IO", f"{t_exec:.2f} s")
    m2.metric("Empreinte RAM Initiale", f"{mem_init:.1f} MB")
    m3.metric("Empreinte RAM Optimisée", f"{mem_fin:.1f} MB", delta=f"-{((mem_init-mem_fin)/mem_init)*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    st.subheader("1.2. Cartographie de la Sparsité (Missing Values)")
    st.markdown("Analyse de la complétude du dataset après conversion des anomalies sentinelles (`-1`). Crucial pour déterminer la stratégie d'imputation (SimpleImputer, KNNImputer).")

    def calculate_sparsity(df):
        missing = (df.isna().sum() / len(df)) * 100
        return missing[missing > 0].sort_values(ascending=False)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.write("**Lieux (Sparsité > 0%)**")
        st.dataframe(calculate_sparsity(raw_data["lieux"]).to_frame(name="% NaN"), use_container_width=True)
    with col_s2:
        st.write("**Usagers (Sparsité > 0%)**")
        st.dataframe(calculate_sparsity(raw_data["usagers"]).to_frame(name="% NaN"), use_container_width=True)
    with col_s3:
        st.write("**Véhicules (Sparsité > 0%)**")
        st.dataframe(calculate_sparsity(raw_data["vehicules"]).to_frame(name="% NaN"), use_container_width=True)

# ----------------------------------------------------------------------
# ONGLET 2 : ANALYSE UNIVARIÉE (STATISTIQUES DESCRIPTIVES)
# ----------------------------------------------------------------------
with tab_univariate:
    st.header("2. Analyse Descriptive Univariée")
    st.markdown("Étude des distributions marginales pour identifier les asymétries (Skewness) et les valeurs aberrantes (Outliers).")

    c_uni1, c_uni2 = st.columns(2)

    with c_uni1:
        st.subheader("Distribution Démographique (Âge)")
        fig_age = px.histogram(
            dim_usg, x="age", nbins=80,
            title="Histogramme & Boxplot de l'Âge",
            marginal="box", color_discrete_sequence=["#2ca02c"]
        )
        # Indicateurs statistiques
        mean_age = dim_usg["age"].mean()
        median_age = dim_usg["age"].median()
        fig_age.add_vline(x=mean_age, line_dash="dash", line_color="red", annotation_text="Moyenne")
        fig_age.add_vline(x=median_age, line_dash="dot", line_color="blue", annotation_text="Médiane")
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption(f"Statistiques : Moyenne = {mean_age:.1f} ans | Médiane = {median_age:.1f} ans. La présence de valeurs négatives ou > 100 ans a été identifiée.")

    with c_uni2:
        st.subheader("Distribution des Vitesses Autorisées (VMA)")
        vma_counts = fact_acc["vma"].value_counts().reset_index()
        vma_counts.columns = ["Vitesse", "Fréquence"]
        vma_counts = vma_counts[vma_counts["Vitesse"] <= 130] # Filtrage des aberrations évidentes

        fig_vma = px.bar(
            vma_counts.head(10), x="Vitesse", y="Fréquence",
            title="Top 10 des VMA les plus fréquentes (≤ 130 km/h)",
            text_auto=".2s", color="Vitesse", color_continuous_scale="Blues"
        )
        fig_vma.update_xaxes(type='category')
        st.plotly_chart(fig_vma, use_container_width=True)

# ----------------------------------------------------------------------
# ONGLET 3 : ANALYSE MULTIVARIÉE & CORRÉLATIONS
# ----------------------------------------------------------------------
with tab_multivariate:
    st.header("3. Interactions des Features et Modélisation")

    st.subheader("3.1. Matrice de Corrélation (Spearman)")
    st.markdown("Utilisation de la corrélation de Spearman, plus robuste aux distributions non normales et adaptée aux variables ordinales (Luminosité, Météo, Sévérité).")

    features_numeriques = ["heure", "latitude", "longitude", "lum", "atm", "vma", "nb_vehicules", "nb_personnes", "severity_index"]
    df_corr = fact_acc[features_numeriques].dropna().corr(method='spearman')

    fig_corr = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns, y=df_corr.index,
        colorscale='RdBu_r', zmin=-1, zmax=1,
        text=np.round(df_corr.values, 2), texttemplate="%{text}",
        hoverinfo="z"
    ))
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("3.2. Analyse des Facteurs de Sévérité (Violin Plots)")
    c_vio1, c_vio2 = st.columns(2)
    with c_vio1:
        fig_vio_veh = px.violin(fact_acc.dropna(subset=['severity_index']), x="Sévérité", y="nb_vehicules", box=True, title="Nombre de véhicules vs Gravité")
        st.plotly_chart(fig_vio_veh, use_container_width=True)
    with c_vio2:
        fig_vio_usg = px.violin(fact_acc.dropna(subset=['severity_index']), x="Sévérité", y="nb_personnes", box=True, title="Nombre de victimes vs Gravité")
        st.plotly_chart(fig_vio_usg, use_container_width=True)

# ----------------------------------------------------------------------
# ONGLET 4 : DASHBOARD BUSINESS INTELLIGENCE
# ----------------------------------------------------------------------
with tab_dashboard:
    st.sidebar.header("Moteur de Rendu (Filtres)")

    f_gravite = st.sidebar.multiselect("Niveau de Sévérité", options=GRAV_LABELS.values(), default=list(GRAV_LABELS.values()))
    f_route = st.sidebar.multiselect("Typologie du Réseau", options=CATR_LABELS.values(), default=list(CATR_LABELS.values()))
    f_jour = st.sidebar.multiselect("Jour de la Semaine", options=list(JOURS_FR.values()), default=list(JOURS_FR.values()))

    df_bi = fact_acc[
        (fact_acc["Sévérité"].isin(f_gravite)) &
        (fact_acc["Type_Route"].isin(f_route)) &
        (fact_acc["jour_semaine"].isin(f_jour))
    ]

    st.header("4. Dashboard Décisionnel Interactif")

    # Indicateurs Macro
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accidents Comptabilisés", f"{len(df_bi):,}")
    k2.metric("Personnes Impliquées", f"{df_bi['nb_personnes'].sum():,.0f}")
    k3.metric("Véhicules Impliqués", f"{df_bi['nb_vehicules'].sum():,.0f}")
    tx_mort = (df_bi["severity_index"] == 2).mean() * 100 if len(df_bi) > 0 else 0
    k4.metric("Taux de Létalité", f"{tx_mort:.2f}%")

    st.markdown("---")

    col_g1, col_g2 = st.columns([1.5, 1])

    with col_g1:
        st.subheader("Concentration Temporelle (Heatmap Croisée)")
        # Génération d'une matrice croisée Jours x Heures
        pivot_t = df_bi.pivot_table(index="jour_semaine", columns="heure", values="Num_Acc", aggfunc="count").fillna(0)
        jours_ordonnes = list(JOURS_FR.values())
        pivot_t = pivot_t.reindex(jours_ordonnes)

        fig_heat_time = px.imshow(
            pivot_t,
            labels=dict(x="Heure de la journée", y="Jour de la semaine", color="Volumétrie"),
            x=pivot_t.columns, y=pivot_t.index,
            color_continuous_scale="Magma", aspect="auto"
        )
        st.plotly_chart(fig_heat_time, use_container_width=True)

    with col_g2:
        st.subheader("Répartition des Véhicules")
        veh_bi = dim_veh[dim_veh["Num_Acc"].isin(df_bi["Num_Acc"])]
        veh_counts = veh_bi["Categorie"].value_counts().reset_index()
        veh_counts.columns = ["Catégorie", "Volume"]
        fig_pie = px.pie(veh_counts.head(8), names="Catégorie", values="Volume", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    st.subheader("Analyse Spatiale et Clustering (France Métropolitaine)")
    st.markdown("Cartographie des points de collision. Affichage limité à un échantillon aléatoire (Max 10 000 points) pour garantir les performances du navigateur web.")

    map_df = df_bi.dropna(subset=["latitude", "longitude"])
    map_df = map_df[(map_df["latitude"].between(41, 51.5)) & (map_df["longitude"].between(-5.5, 10))]

    if not map_df.empty:
        fig_map = px.scatter_map(
            map_df.sample(min(10000, len(map_df)), random_state=42),
            lat="latitude", lon="longitude",
            color="Sévérité", size="nb_personnes",
            color_discrete_map={"Indemne": "green", "Blessé léger": "yellow", "Blessé hospitalisé": "orange", "Tué": "red", "Inconnu": "gray"},
            zoom=5, height=600,
            hover_data=["Type_Route", "heure", "Météo"],
            map_style="carto-positron"
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Volume de données géospatiales insuffisant avec les filtres actuels.")