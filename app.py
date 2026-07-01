"""
Tableau de bord Streamlit Analytique - Accidentologie France (2024)
-------------------------------------------------------------------
Pipeline Medallion (Bronze -> Silver -> Gold)
Comprend une Analyse Exploratoire de Données (EDA) approfondie et un Dashboard BI.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Data Science - Analyse Sécurité Routière",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Dictionnaires de référence (Features Categoriques)
# ----------------------------------------------------------------------
LUM_LABELS = {1: "Plein jour", 2: "Crépuscule/Aube", 3: "Nuit (sans éclairage)", 4: "Nuit (éclairage éteint)", 5: "Nuit (éclairage allumé)"}
ATM_LABELS = {1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige/Grêle", 5: "Brouillard/Fumée", 6: "Vent fort", 7: "Temps éblouissant", 8: "Couvert", 9: "Autre"}
CATR_LABELS = {1: "Autoroute", 2: "Route Nationale", 3: "Route Départementale", 4: "Voie Communale", 5: "Hors réseau public", 6: "Parc de stationnement", 9: "Autre"}
CATU_LABELS = {1: "Conducteur", 2: "Passager", 3: "Piéton"}
GRAV_LABELS = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
CATV_LABELS = {1: "Vélo", 2: "Cyclomoteur", 7: "Voiture", 10: "Véhicule utilitaire", 13: "Poids lourd (<7.5t)", 14: "Poids lourd (>=7.5t)", 30: "Scooter", 33: "Moto", 37: "Bus", 38: "Autocar"}
SEXE_LABELS = {1: "Masculin", 2: "Féminin", -1: "Non renseigné"}

# ======================================================================
# COUCHE BRONZE : EXTRACTION
# ======================================================================
@st.cache_data(show_spinner="Extraction de la couche Bronze...")
def load_raw_data():
    return {
        "caracteristiques": pd.read_csv("caract-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "lieux": pd.read_csv("lieux-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "usagers": pd.read_csv("usagers-2024.csv", sep=";", encoding="utf-8", low_memory=False),
        "vehicules": pd.read_csv("vehicules-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    }

# ======================================================================
# COUCHE SILVER & GOLD : TRANSFORMATION ET MODÉLISATION
# ======================================================================
@st.cache_data(show_spinner="Ingénierie des features (Silver -> Gold)...")
def build_analytical_model(_raw):
    caract, lieux, usagers, vehicules = _raw["caracteristiques"].copy(), _raw["lieux"].copy(), _raw["usagers"].copy(), _raw["vehicules"].copy()

    # --- SILVER : DATA CLEANSING ---
    caract["date"] = pd.to_datetime(caract["an"].astype(str) + "-" + caract["mois"].astype(str) + "-" + caract["jour"].astype(str), errors="coerce")
    caract["datetime"] = pd.to_datetime(caract["date"].dt.strftime("%Y-%m-%d") + " " + caract["hrmn"], errors="coerce")
    caract["latitude"] = pd.to_numeric(caract["lat"].astype(str).str.replace(",", "."), errors="coerce")
    caract["longitude"] = pd.to_numeric(caract["long"].astype(str).str.replace(",", "."), errors="coerce")

    # Remplacement global de la sentinelle -1 par NaN sur les valeurs numériques
    for df in [lieux, usagers, vehicules]:
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col not in ["Num_Acc", "id_usager", "id_vehicule"]:
                df.loc[df[col] == -1, col] = np.nan

    lieux = lieux.drop_duplicates(subset=["Num_Acc"], keep="first")

    # Feature Engineering
    usagers["age"] = 2024 - pd.to_numeric(usagers["an_nais"], errors="coerce")
    caract["heure"] = caract["datetime"].dt.hour
    caract["jour_semaine"] = caract["datetime"].dt.day_name(locale="fr_FR.utf8" if hasattr(caract["datetime"].dt, "day_name") else None)
    caract["time_of_day"] = pd.cut(caract["heure"], bins=[0, 6, 12, 18, 24], labels=["Nuit", "Matin", "Après-midi", "Soirée"], right=False)

    worst_injury = usagers.groupby("Num_Acc")["grav"].max().rename("severity_index")
    caract = caract.merge(worst_injury, on="Num_Acc", how="left")

    # --- GOLD : FACT & DIMENSIONS ---
    fact = caract[["Num_Acc", "datetime", "heure", "time_of_day", "jour_semaine", "latitude", "longitude", "lum", "atm", "severity_index"]].copy()
    fact = fact.merge(lieux[["Num_Acc", "catr", "vma"]], on="Num_Acc", how="left")

    nb_veh = vehicules.groupby("Num_Acc").size().rename("nb_vehicules")
    nb_usg = usagers.groupby("Num_Acc").size().rename("nb_personnes")
    fact = fact.merge(nb_veh, on="Num_Acc", how="left").merge(nb_usg, on="Num_Acc", how="left")

    fact["Luminosité"] = fact["lum"].map(LUM_LABELS).fillna("Inconnu")
    fact["Météo"] = fact["atm"].map(ATM_LABELS).fillna("Inconnu")
    fact["Type_Route"] = fact["catr"].map(CATR_LABELS).fillna("Inconnu")
    fact["Sévérité"] = fact["severity_index"].map(GRAV_LABELS).fillna("Inconnu")

    dim_veh = vehicules[["id_vehicule", "Num_Acc", "catv", "obs", "choc"]].copy()
    dim_veh["Categorie"] = dim_veh["catv"].map(CATV_LABELS).fillna("Autre")

    dim_usg = usagers[["id_usager", "Num_Acc", "catu", "grav", "age", "sexe", "trajet"]].copy()
    dim_usg["Role"] = dim_usg["catu"].map(CATU_LABELS).fillna("Inconnu")
    dim_usg["Sexe_Label"] = dim_usg["sexe"].map(SEXE_LABELS).fillna("Inconnu")

    return fact, dim_veh, dim_usg, caract, lieux, usagers, vehicules

# ======================================================================
# INTERFACE
# ======================================================================
st.title("🚦 Analyse Approfondie de l'Accidentologie Française")
st.markdown("Ce tableau de bord présente une Analyse Exploratoire des Données (EDA) rigoureuse en préparation d'une modélisation prédictive, suivie d'un outil de Business Intelligence interactif.")

try:
    raw_data = load_raw_data()
    fact_acc, dim_veh, dim_usg, df_caract, df_lieux, df_usagers, df_vehicules = build_analytical_model(raw_data)
except FileNotFoundError:
    st.error("⚠️ Fichiers CSV introuvables à la racine du projet.")
    st.stop()

tab_eda, tab_dashboard = st.tabs(["🔬 Analyse Exploratoire (Data Quality & Profiling)", "📊 Dashboard BI (Gold Layer)"])

# ----------------------------------------------------------------------
# ONGLET 1 : DATA QUALITY & EDA (Niveau Machine Learning)
# ----------------------------------------------------------------------
with tab_eda:
    st.header("1. Audit de la Qualité des Données (Couche Bronze)")

    # 1.1 Volumétrie et intégrité
    st.subheader("1.1. Volumétrie et Intégrité Référentielle")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accidents (Caractéristiques)", f"{len(df_caract):,}")
    col2.metric("Lieux documentés", f"{len(raw_data['lieux']):,}")
    col3.metric("Usagers impliqués", f"{len(df_usagers):,}")
    col4.metric("Véhicules impliqués", f"{len(df_vehicules):,}")

    st.info(f"**Analyse Structurelle :** La table `lieux` brute contient {len(raw_data['lieux'])} lignes pour {raw_data['lieux']['Num_Acc'].nunique()} accidents uniques. Ce doublonnage structurel a été corrigé dans la couche Silver (déduplication) pour éviter une explosion des jointures.")

    st.markdown("---")

    # 1.2 Missing Data Heatmap
    st.subheader("1.2. Cartographie des Valeurs Manquantes (Missing Data)")
    st.markdown("Étude de la complétude post-nettoyage (les sentinelles `-1` ont été converties en vrais `NaN`). Les features avec plus de 30% de valeurs manquantes nécessiteront une stratégie d'imputation lourde (KNN, MICE) ou devront être écartées.")

    def plot_missing_values(df, title):
        missing = (df.isna().sum() / len(df)) * 100
        missing = missing[missing > 0].sort_values(ascending=True)
        if len(missing) == 0: return None
        fig = px.bar(missing, orientation='h', title=title, labels={'value': '% Manquant', 'index': 'Features'})
        fig.update_layout(height=150 + len(missing)*20, showlegend=False)
        return fig

    c_miss1, c_miss2 = st.columns(2)
    with c_miss1:
        st.plotly_chart(plot_missing_values(df_lieux, "Table Lieux : Taux de NaN"), use_container_width=True)
    with c_miss2:
        st.plotly_chart(plot_missing_values(df_usagers, "Table Usagers : Taux de NaN"), use_container_width=True)

    st.markdown("---")

    # 1.3 Analyse Univariée et Outliers
    st.subheader("1.3. Analyse Univariée et Détection d'Outliers")
    st.markdown("Étude des distributions pour détecter les anomalies statistiques aberrantes avant l'entraînement d'algorithmes de Machine Learning.")

    out1, out2 = st.columns(2)
    with out1:
        # Distribution des âges
        fig_age = px.histogram(dim_usg, x="age", nbins=100, title="Distribution de l'Âge des Usagers",
                               marginal="box", color_discrete_sequence=["#1f77b4"])
        fig_age.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Anomalie (<0)")
        fig_age.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Anomalie (>100)")
        st.plotly_chart(fig_age, use_container_width=True)

    with out2:
        # VMA (Vitesse Maximale Autorisée)
        fig_vma = px.box(fact_acc, x="vma", title="Dispersion des Vitesses Maximales Autorisées (VMA)", points="outliers")
        st.plotly_chart(fig_vma, use_container_width=True)
        st.caption("On observe des vitesses à des valeurs non standards (ex: 1 km/h, 999 km/h) qui devront être clippées ou remplacées par le mode.")

    st.markdown("---")

    # 1.4 Analyse Multivariée & Corrélations
    st.subheader("1.4. Analyse Bivariée et Matrice de Corrélation")
    st.markdown("Étude des corrélations linéaires (Pearson) entre les variables numériques continues et ordinales.")

    # Préparation d'un df numérique pour la corrélation
    corr_cols = ["heure", "latitude", "longitude", "lum", "atm", "severity_index", "nb_vehicules", "nb_personnes", "vma"]
    df_corr = fact_acc[corr_cols].dropna().corr(method='spearman') # Spearman est plus adapté car variables ordinales

    fig_corr = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns,
        y=df_corr.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(df_corr.values, 2),
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    fig_corr.update_layout(title="Matrice de corrélation de Spearman", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------------------------------------------------
# ONGLET 2 : DASHBOARD ANALYTIQUE BI
# ----------------------------------------------------------------------
with tab_dashboard:
    st.sidebar.header("Moteur de Filtrage")

    s_grav = st.sidebar.multiselect("Sévérité (Index)", options=GRAV_LABELS.values(), default=list(GRAV_LABELS.values()))
    s_route = st.sidebar.multiselect("Type de Route", options=CATR_LABELS.values(), default=list(CATR_LABELS.values()))

    # Application des filtres métier
    df_dash = fact_acc[
        (fact_acc["Sévérité"].isin(s_grav)) &
        (fact_acc["Type_Route"].isin(s_route))
    ]

    st.header("Tableau de Bord Accidentologique")

    # High-level KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Accidents Filtrés", f"{len(df_dash):,}", help="Nombre total de sinistres pour cette sélection")
    kpi2.metric("Victimes Impliquées", f"{df_dash['nb_personnes'].sum():,.0f}")

    # Calcul de la gravité moyenne
    avg_sev = df_dash['severity_index'].mean()
    kpi3.metric("Indice de Gravité Moyen", f"{avg_sev:.2f} / 4")

    pct_mortalite = (df_dash["severity_index"] == 2).mean() * 100 if len(df_dash) > 0 else 0
    kpi4.metric("Taux d'Accidents Mortels", f"{pct_mortalite:.2f}%")

    st.markdown("---")

    # Ligne 1 : Temporalité
    c_time1, c_time2 = st.columns([2, 1])

    with c_time1:
        st.subheader("Concentration Temporelle (Heatmap Jour/Heure)")
        # Préparation des données pour la heatmap temporelle
        pivot_time = df_dash.pivot_table(index="jour_semaine", columns="heure", values="Num_Acc", aggfunc="count").fillna(0)
        # Ordonner les jours de la semaine
        jours_ordre = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot_time = pivot_time.reindex(jours_ordre)

        fig_time_heat = px.imshow(pivot_time, labels=dict(x="Heure de la journée", y="Jour", color="Accidents"),
                                  x=pivot_time.columns, y=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
                                  color_continuous_scale="Viridis", aspect="auto")
        st.plotly_chart(fig_time_heat, use_container_width=True)

    with c_time2:
        st.subheader("Distribution par Sexe et Rôle")
        # Jointure pour récupérer les infos usagers sur le set filtré
        usg_dash = dim_usg[dim_usg["Num_Acc"].isin(df_dash["Num_Acc"])]
        fig_sexe = px.histogram(usg_dash, x="Role", color="Sexe_Label", barmode="group",
                                color_discrete_map={"Masculin": "#1f77b4", "Féminin": "#e377c2", "Non renseigné": "gray"})
        st.plotly_chart(fig_sexe, use_container_width=True)

    st.markdown("---")

    # Ligne 2 : Analyse Topologique et Cartographie
    c_map1, c_map2 = st.columns([1, 1])

    with c_map1:
        st.subheader("Analyse Hiérarchique : Route et Conditions")
        st.markdown("Décomposition des accidents : Réseau -> Météo -> Sévérité")
        # Sunburst chart très apprécié en analyse BI
        sun_data = df_dash.groupby(["Type_Route", "Météo", "Sévérité"]).size().reset_index(name='count')
        # On filtre les très petites catégories pour la lisibilité
        sun_data = sun_data[sun_data['count'] > 50]
        fig_sun = px.sunburst(sun_data, path=['Type_Route', 'Météo', 'Sévérité'], values='count',
                              color='Sévérité', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sun.update_layout(height=500)
        st.plotly_chart(fig_sun, use_container_width=True)

    with c_map2:
        st.subheader("Cartographie Spatiale")
        map_data = df_dash.dropna(subset=["latitude", "longitude"])
        # Centrage sur la France Métropolitaine pour éviter un zoom mondial à cause des DROM-COM
        map_data = map_data[(map_data["latitude"].between(41, 51.5)) & (map_data["longitude"].between(-5.5, 10))]

        if not map_data.empty:
            fig_map = px.scatter_map(
                map_data.sample(min(5000, len(map_data)), random_state=42),
                lat="latitude", lon="longitude", color="Sévérité",
                zoom=4.5, height=500, hover_data=["Type_Route", "heure", "Météo"]
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Données géospatiales insuffisantes pour l'affichage avec ces filtres.")