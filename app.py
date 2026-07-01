"""
Tableau de bord Streamlit - Sécurité Routière France (2024)
------------------------------------------------------
Architecture Medallion (Bronze -> Silver -> Gold)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sécurité Routière - Qualité & Analyse",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Analyse des Accidents de la Route en France (2024)")
st.caption("Pipeline de données : Évaluation de la qualité (Bronze) ➔ Tableau de bord analytique (Gold)")

# ----------------------------------------------------------------------
# Dictionnaires de traduction pour rendre les données lisibles
# ----------------------------------------------------------------------
LUM_LABELS = {
    1: "Plein jour", 2: "Crépuscule/Aube", 3: "Nuit (sans éclairage)",
    4: "Nuit (éclairage éteint)", 5: "Nuit (éclairage allumé)",
}
ATM_LABELS = {
    1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige/Grêle",
    5: "Brouillard/Fumée", 6: "Vent fort", 7: "Temps éblouissant",
    8: "Couvert", 9: "Autre",
}
CATR_LABELS = {
    1: "Autoroute", 2: "Route Nationale", 3: "Route Départementale",
    4: "Voie Communale", 5: "Hors réseau public", 6: "Parc de stationnement", 9: "Autre",
}
CATU_LABELS = {1: "Conducteur", 2: "Passager", 3: "Piéton"}
GRAV_LABELS = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
CATV_LABELS = {
    1: "Vélo", 2: "Cyclomoteur", 7: "Voiture", 10: "Véhicule utilitaire",
    13: "Poids lourd (<7.5t)", 14: "Poids lourd (>=7.5t)", 30: "Scooter", 33: "Moto",
    37: "Bus", 38: "Autocar",
}

# ----------------------------------------------------------------------
# Couche BRONZE : Chargement des données brutes
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des fichiers bruts...")
def load_raw():
    caract = pd.read_csv("caract-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    lieux = pd.read_csv("lieux-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    usagers = pd.read_csv("usagers-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    vehicules = pd.read_csv("vehicules-2024.csv", sep=";", encoding="utf-8", low_memory=False)
    return {"caractéristiques": caract, "lieux": lieux, "usagers": usagers, "véhicules": vehicules}

# ----------------------------------------------------------------------
# Profilage & Qualité des données (Partie 1 du TP)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Analyse de la qualité des données en cours...")
def compute_quality_report(_raw):
    report = {}
    missing_tables = {}
    completeness_scores = {}

    for name, df in _raw.items():
        n = len(df)

        # Calcul des cellules purement vides (NaN)
        true_na_pct = (df.isna().sum() / n * 100).round(2)

        # Calcul des cellules "Non renseignées" (Code -1)
        sentinel_pct = {}
        for col in df.columns:
            count = (df[col].astype(str).str.strip() == "-1").sum()
            sentinel_pct[col] = round(count / n * 100, 2)

        combined = pd.DataFrame({
            "Valeurs Absentes (%)": true_na_pct,
            "Non Renseigné (Code -1) (%)": pd.Series(sentinel_pct)
        }).fillna(0)

        combined["Total Manquant (%)"] = combined["Valeurs Absentes (%)"] + combined["Non Renseigné (Code -1) (%)"]

        # Score de complétude de la table (100% - moyenne des manques)
        completeness_scores[name] = round(100 - combined["Total Manquant (%)"].mean(), 1)

        # On ne garde que les colonnes qui ont des problèmes
        combined = combined[combined["Total Manquant (%)"] > 0]
        combined = combined.sort_values("Total Manquant (%)", ascending=False)
        missing_tables[name] = combined

    report["missing_tables"] = missing_tables
    report["completeness_scores"] = completeness_scores

    # Détection des doublons
    report["exact_duplicates"] = {name: int(df.duplicated().sum()) for name, df in _raw.items()}

    # Anomalie structurelle sur la table 'lieux'
    lieux = _raw["lieux"]
    acc_count = lieux["Num_Acc"].value_counts()
    multi = acc_count[acc_count > 1]
    report["lieux_total_rows"] = len(lieux)
    report["lieux_unique_accidents"] = lieux["Num_Acc"].nunique()
    report["lieux_accidents_with_many_rows"] = len(multi)

    # Intégrité référentielle
    caract_ids = set(_raw["caractéristiques"]["Num_Acc"])
    integrity = {}
    for name in ["lieux", "usagers", "véhicules"]:
        orphan = set(_raw[name]["Num_Acc"]) - caract_ids
        integrity[name] = len(orphan)
    report["referential_integrity"] = integrity

    return report

# ----------------------------------------------------------------------
# Couches SILVER + GOLD : Transformations et Modélisation
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Construction des couches Silver et Gold...")
def build_gold_data(_raw):
    caract = _raw["caractéristiques"].copy()
    lieux = _raw["lieux"].copy()
    usagers = _raw["usagers"].copy()
    vehicules = _raw["véhicules"].copy()

    # SILVER : Nettoyage et normalisation
    caract["date"] = pd.to_datetime(
        caract["an"].astype(str) + "-" + caract["mois"].astype(str) + "-" + caract["jour"].astype(str),
        format="%Y-%m-%d", errors="coerce"
    )
    caract["datetime"] = pd.to_datetime(
        caract["date"].dt.strftime("%Y-%m-%d") + " " + caract["hrmn"], errors="coerce"
    )
    caract["latitude"] = caract["lat"].astype(str).str.replace(",", ".").astype(float)
    caract["longitude"] = caract["long"].astype(str).str.replace(",", ".").astype(float)

    # Remplacement des codes -1 par de vrais NaN pour l'analyse
    for df in [lieux, usagers, vehicules]:
        for col in df.columns:
            if col not in ["Num_Acc", "id_usager", "id_vehicule"] and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace(-1, np.nan)

    # Correction du problème de doublons dans 'lieux' (Deduplication)
    lieux = lieux.sort_values("Num_Acc").drop_duplicates(subset="Num_Acc", keep="first")

    # Calcul de l'index de sévérité de l'accident (la pire blessure parmi les usagers)
    worst_injury = usagers.groupby("Num_Acc")["grav"].max().rename("severity_index")
    caract = caract.merge(worst_injury, on="Num_Acc", how="left")

    def get_time_of_day(dt):
        if pd.isna(dt): return np.nan
        hour = dt.hour
        if 6 <= hour < 12: return "Matin"
        elif 12 <= hour < 18: return "Après-midi"
        elif 18 <= hour < 23: return "Soirée"
        else: return "Nuit"

    caract["time_of_day"] = caract["datetime"].apply(get_time_of_day)
    usagers["age"] = 2024 - usagers["an_nais"]

    # GOLD : Table de faits (Accidents)
    fact = caract[[
        "Num_Acc", "datetime", "time_of_day", "latitude", "longitude",
        "lum", "atm", "col", "severity_index",
    ]].copy()

    fact = fact.merge(lieux[["Num_Acc", "catr", "nbv", "vma"]], on="Num_Acc", how="left")

    nb_vehicles = vehicules.groupby("Num_Acc").size().rename("nb_vehicles")
    nb_persons = usagers.groupby("Num_Acc").size().rename("nb_persons")
    fact = fact.merge(nb_vehicles, on="Num_Acc", how="left")
    fact = fact.merge(nb_persons, on="Num_Acc", how="left")

    # Ajout des labels lisibles
    fact["Luminosité"] = fact["lum"].map(LUM_LABELS).fillna("Inconnu")
    fact["Météo"] = fact["atm"].map(ATM_LABELS).fillna("Inconnu")
    fact["Type de Route"] = fact["catr"].map(CATR_LABELS).fillna("Inconnu")
    fact["Sévérité"] = fact["severity_index"].map(GRAV_LABELS).fillna("Inconnu")

    # GOLD : Tables de dimensions
    dim_vehicle = vehicules[["id_vehicule", "Num_Acc", "catv"]].drop_duplicates()
    dim_vehicle["Catégorie Véhicule"] = dim_vehicle["catv"].map(CATV_LABELS).fillna("Autre")

    dim_person = usagers[["id_usager", "Num_Acc", "catu", "grav", "sexe", "age"]].drop_duplicates()
    dim_person["Rôle"] = dim_person["catu"].map(CATU_LABELS).fillna("Inconnu")
    dim_person["Gravité Blessure"] = dim_person["grav"].map(GRAV_LABELS).fillna("Inconnu")

    return fact, dim_vehicle, dim_person

# ----------------------------------------------------------------------
# Exécution principale
# ----------------------------------------------------------------------
try:
    raw = load_raw()
except FileNotFoundError:
    st.error("Fichiers introuvables. Placez caract-2024.csv, lieux-2024.csv, usagers-2024.csv et vehicules-2024.csv dans le même dossier.")
    st.stop()

quality = compute_quality_report(raw)
fact_accidents, dim_vehicle, dim_person = build_gold_data(raw)

# ----------------------------------------------------------------------
# Interface Graphique (Onglets)
# ----------------------------------------------------------------------
tab_quality, tab_dashboard = st.tabs(["📋 Qualité des Données (Audit)", "📊 Tableau de Bord (Analyse)"])

# ========================================================================
# ONGLET 1 : QUALITÉ DES DONNÉES
# ========================================================================
with tab_quality:
    st.header("Audit de Qualité des Données Ouvertes")
    st.markdown("Cette section évalue l'état brut des données (Couche Bronze) avant leur nettoyage. L'objectif est d'identifier les lacunes qui pourraient fausser nos analyses futures.")

    # Scores de complétude
    st.subheader("Score de Santé par Table")
    st.caption("Le score représente le pourcentage d'informations correctement remplies par rapport au total attendu.")
    score_cols = st.columns(4)
    for c, (name, score) in zip(score_cols, quality["completeness_scores"].items()):
        c.metric(name.capitalize(), f"{score}%", f"{len(raw[name]):,} lignes", delta_color="off")

    st.markdown("---")

    # Explication claire des valeurs manquantes
    st.subheader("Analyse de la Complétude (Données Manquantes)")
    st.info("💡 **Comment lire ceci ?** Dans ce jeu de données, l'information peut manquer de deux façons : soit la case est purement **absente (vide)**, soit l'officier de police a explicitement indiqué que l'information n'était **pas renseignée (Code d'erreur -1)**. Le tableau ci-dessous combine ces deux anomalies pour afficher le vrai pourcentage de données inutilisables.")

    mq_cols = st.columns(4)
    for c, (name, tbl) in zip(mq_cols, quality["missing_tables"].items()):
        with c:
            st.markdown(f"**{name.capitalize()}**")
            if tbl.empty:
                st.write("✅ Aucune donnée manquante.")
            else:
                st.dataframe(tbl, height=250)

    # Graphique global des pires colonnes
    all_missing = []
    for name, tbl in quality["missing_tables"].items():
        for col in tbl.index:
            all_missing.append({"Table": name, "Colonne": col, "Données Inutilisables (%)": tbl.loc[col, "Total Manquant (%)"]})

    missing_df = pd.DataFrame(all_missing).sort_values("Données Inutilisables (%)", ascending=False).head(10)
    fig_missing = px.bar(
        missing_df, x="Données Inutilisables (%)", y="Colonne", color="Table", orientation="h",
        title="Top 10 des colonnes les moins fiables",
        text_auto=".1f"
    )
    fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_missing, use_container_width=True)

    st.markdown("---")

    # Section intégrité et doublons clarifiée
    st.subheader("Intégrité Structurelle")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Cohérence entre les tables (Intégrité Référentielle)**")
        st.markdown("Vérifie si tous les accidents mentionnés dans les tables secondaires existent bien dans la table principale.")
        ref_df = pd.DataFrame({
            "Table": list(quality["referential_integrity"].keys()),
            "Accidents Orphelins": list(quality["referential_integrity"].values())
        })
        st.dataframe(ref_df, hide_index=True)
        if all(v == 0 for v in quality["referential_integrity"].values()):
            st.success("Parfait : Tous les identifiants correspondent, aucune donnée orpheline.")

    with d2:
        st.markdown("**Problème critique détecté dans la table `lieux`**")
        st.error(
            f"⚠️ La table contient **{quality['lieux_total_rows']:,} lignes** pour seulement "
            f"**{quality['lieux_unique_accidents']:,} accidents uniques**.\n\n"
            f"Il y a des doublons structurels ({quality['lieux_accidents_with_many_rows']:,} accidents ont plusieurs lignes). "
            "Si on ne nettoie pas cela, les jointures vont multiplier artificiellement le nombre d'accidents dans le tableau de bord."
        )

# ========================================================================
# ONGLET 2 : DASHBOARD
# ========================================================================
with tab_dashboard:
    st.sidebar.header("Filtres Analytiques")

    time_options = ["Matin", "Après-midi", "Soirée", "Nuit"]
    selected_times = st.sidebar.multiselect("Moment de la journée", time_options, default=time_options)

    road_options = sorted(fact_accidents["Type de Route"].dropna().unique())
    selected_roads = st.sidebar.multiselect("Type de Route", road_options, default=road_options)

    severity_options = sorted(fact_accidents["Sévérité"].dropna().unique())
    selected_severity = st.sidebar.multiselect("Gravité maximale", severity_options, default=severity_options)

    # Application des filtres
    filtered = fact_accidents[
        fact_accidents["time_of_day"].isin(selected_times)
        & fact_accidents["Type de Route"].isin(selected_roads)
        & fact_accidents["Sévérité"].isin(selected_severity)
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(filtered):,} accidents correspondent à ces critères.")

    st.header("Vue d'ensemble de l'Accidentologie")

    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    total_accidents = len(filtered)
    total_persons = int(filtered["nb_persons"].sum()) if total_accidents > 0 else 0
    total_vehicles = int(filtered["nb_vehicles"].sum()) if total_accidents > 0 else 0
    pct_killed = (filtered["severity_index"] == 2).mean() * 100 if total_accidents > 0 else 0

    col1.metric("Accidents enregistrés", f"{total_accidents:,}")
    col2.metric("Personnes impliquées", f"{total_persons:,}")
    col3.metric("Véhicules impliqués", f"{total_vehicles:,}")
    col4.metric("Accidents mortels", f"{pct_killed:.1f}%")

    st.markdown("---")

    # Analyse approfondie
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Analyse de Sévérité par Météo")
        st.markdown("Identifie si les conditions atmosphériques aggravent la mortalité/blessure.")
        avg_severity_atm = filtered.groupby("Météo")["severity_index"].mean().reset_index()
        fig1 = px.bar(
            avg_severity_atm.sort_values("severity_index", ascending=False),
            x="Météo", y="severity_index",
            color="severity_index",
            labels={"severity_index": "Indice de gravité moyen (1-4)"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Répartition par Type de Route")
        by_road = filtered["Type de Route"].value_counts().reset_index()
        by_road.columns = ["Type de Route", "Nombre d'accidents"]
        fig2 = px.bar(by_road, x="Type de Route", y="Nombre d'accidents", color="Type de Route")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Cartographie
    st.subheader("Cartographie des incidents")
    map_data = filtered.dropna(subset=["latitude", "longitude"])
    map_data = map_data[
        (map_data["latitude"].between(41, 51.5)) & (map_data["longitude"].between(-5.5, 10))
    ]
    if len(map_data) > 0:
        fig3 = px.scatter_map(
            map_data.sample(min(4000, len(map_data)), random_state=42),
            lat="latitude", lon="longitude",
            color="Sévérité",
            hover_data=["time_of_day", "Type de Route"],
            zoom=4.5, height=500,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Aucune coordonnée valide pour ces filtres.")