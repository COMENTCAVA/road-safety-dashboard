import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# ======================================================================
# CONFIGURATION & STYLE
# ======================================================================
st.set_page_config(
    page_title="Audit Data Quality - Sécurité Routière",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom pour un rendu professionnel
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric label {
        color: #495057 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #212529 !important;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f1f3f5;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        border-bottom: 2px solid #0d6efd !important;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid;
    }
    .alert-info { background-color: #e7f1ff; border-color: #0d6efd; color: #084298; }
    .alert-warning { background-color: #fff3cd; border-color: #ffc107; color: #664d03; }
    .alert-danger { background-color: #f8d7da; border-color: #dc3545; color: #842029; }
</style>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0d6efd", "#dc3545", "#198754", "#ffc107", "#6f42c1"]

# ======================================================================
# CHARGEMENT DES DONNÉES (COUCHE BRONZE)
# ======================================================================
@st.cache_data(show_spinner="Chargement et audit du schéma brut...")
def load_bronze_data():
    files = {
        "caract": "caract-2024.csv",
        "lieux": "lieux-2024.csv",
        "usagers": "usagers-2024.csv",
        "vehicules": "vehicules-2024.csv"
    }
    data = {}
    for key, file in files.items():
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, sep=";", encoding="utf-8", low_memory=False)
                data[key] = df
            except Exception:
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()
    return data

# ======================================================================
# FONCTIONS D'AUDIT (DATA PROFILING)
# ======================================================================
@st.cache_data
def analyze_sentinels(_df):
    """Détecte les valeurs sentinelles '-1' typiques du jeu de données BAAC."""
    sentinels = {}
    for col in _df.columns:
        if col not in ['Num_Acc', 'id_usager', 'id_vehicule']:
            # Conversion en string pour capturer les -1 numériques et textuels
            count = (_df[col].astype(str).str.strip() == '-1').sum()
            if count > 0:
                sentinels[col] = count
    return pd.Series(sentinels).sort_values(ascending=False)

@st.cache_data
def analyze_gps(_df):
    """Analyse la qualité des coordonnées GPS (problème de la virgule)."""
    if 'lat' not in _df.columns or 'long' not in _df.columns:
        return None

    lat_raw = _df['lat'].astype(str)
    lon_raw = _df['long'].astype(str)

    # Problème de format (virgule au lieu du point)
    comma_lat = lat_raw.str.contains(',').sum()
    comma_lon = lon_raw.str.contains(',').sum()

    # Nettoyage temporaire pour vérification
    lat_clean = pd.to_numeric(lat_raw.str.replace(',', '.'), errors='coerce')
    lon_clean = pd.to_numeric(lon_raw.str.replace(',', '.'), errors='coerce')

    nulls = lat_clean.isna().sum()
    zeros = ((lat_clean == 0) & (lon_clean == 0)).sum()

    # Vérification bounding box France Métropolitaine
    metro_mask = (lat_clean.between(41.0, 51.5)) & (lon_clean.between(-5.5, 10.0))
    valid_metro = (metro_mask & lat_clean.notna() & (lat_clean != 0)).sum()
    outside = (lat_clean.notna() & (lat_clean != 0) & ~metro_mask).sum()

    return {
        "comma_lat": int(comma_lat), "comma_lon": int(comma_lon),
        "nulls": int(nulls), "zeros": int(zeros),
        "valid_metro": int(valid_metro), "outside": int(outside)
    }

@st.cache_data
def analyze_lieux_structure(_df):
    """Détecte les doublons dans la table lieux (1 accident = plusieurs lignes)."""
    if _df.empty or 'Num_Acc' not in _df.columns:
        return None
    counts = _df['Num_Acc'].value_counts()
    multi = counts[counts > 1]
    return {
        "total_accidents": len(counts),
        "multi_rows": len(multi),
        "max_rows": int(counts.max()) if len(counts) > 0 else 0,
        "total_extra_rows": int(multi.sum() - len(multi)) if len(multi) > 0 else 0
    }

@st.cache_data
def analyze_age_logic(_df):
    """Vérifie la cohérence de l'année de naissance pour le calcul d'âge."""
    if _df.empty or 'an_nais' not in _df.columns:
        return None

    an_nais = pd.to_numeric(_df['an_nais'], errors='coerce')
    ages = 2024 - an_nais

    return {
        "negatifs": int((ages < 0).sum()),
        "centenaires": int((ages > 105).sum()),
        "manquants": int(ages.isna().sum()),
        "valides": int(((ages >= 0) & (ages <= 105)).sum())
    }

# ======================================================================
# INTERFACE PRINCIPALE
# ======================================================================
st.title("🔬 Audit Data Quality & Profiling (Couche Bronze)")
st.markdown("""
<div class="alert-box alert-info">
<b>Architecture Medallion - Étape 1 : Audit.</b><br>
Cette application analyse en profondeur les données brutes (Bronze) pour identifier les biais, les erreurs de format et les incohérences structurelles. 
Ces constats justifient les transformations appliquées dans les pipelines <b>Silver</b> (Nettoyage) et <b>Gold</b> (Modélisation).
</div>
""")

# Chargement
raw_data = load_bronze_data()

# Sidebar : État des fichiers
with st.sidebar:
    st.header("📂 État des données")
    for name, df in raw_data.items():
        if not df.empty:
            st.success(f"✅ **{name}** : {len(df):,} lignes")
        else:
            st.error(f"❌ **{name}** : Vide / Introuvable")

    st.markdown("---")
    st.markdown("### 🎯 Objectifs de l'audit")
    st.markdown("""
    1. **Standardisation** (GPS, Dates)
    2. **Sentinelles** (Valeurs `-1`)
    3. **Intégrité** (Doublons `lieux`)
    4. **Logique Métier** (Âges, Référentiel)
    """)

if all(df.empty for df in raw_data.values()):
    st.error("⚠️ Aucun fichier CSV trouvé. Veuillez placer les 4 fichiers à la racine du projet.")
    st.stop()

# Création des onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Volumétrie",
    "🧹 Standardisation & Sentinelles",
    "🔗 Intégrité Structurelle",
    "🧠 Logique Métier",
    "🚀 Pipeline Silver/Gold"
])

# ----------------------------------------------------------------------
# ONGLET 1 : VOLUMETRIE
# ----------------------------------------------------------------------
with tab1:
    st.header("📊 Volumétrie et Empreinte Mémoire")
    st.markdown("Aperçu de la taille des tables et de la cohérence des identifiants (`Num_Acc`).")

    cols = st.columns(4)
    for i, (name, df) in enumerate(raw_data.items()):
        with cols[i]:
            if not df.empty:
                st.metric(f"Table `{name}`", f"{len(df):,}")
                st.caption(f"{df.shape[1]} colonnes | {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            else:
                st.metric(f"Table `{name}`", "N/A")

    st.markdown("### Cohérence des identifiants (Clé primaire)")
    if not raw_data['caract'].empty:
        nb_caract = raw_data['caract']['Num_Acc'].nunique()
        st.info(f"💡 La table `caract` contient **{nb_caract:,}** accidents uniques. Les tables satellites doivent pointer vers ces mêmes IDs.")
    st.markdown("---")

# ----------------------------------------------------------------------
# ONGLET 2 : STANDARDISATION & SENTINELLES
# ----------------------------------------------------------------------
with tab2:
    st.header("🧹 Standardisation & Valeurs Sentinelles")

    c1, c2 = st.columns(2)

    # --- GPS ---
    with c1:
        st.subheader("📍 Qualité de la Géolocalisation (GPS)")
        st.markdown("Dans le format BAAC, les coordonnées sont souvent des textes avec des **virgules** (ex: `48,85`) au lieu de points, ce qui empêche leur utilisation numérique directe.")

        if not raw_data['caract'].empty:
            gps_stats = analyze_gps(raw_data['caract'])
            if gps_stats:
                fig_gps = go.Figure(data=[go.Pie(
                    labels=['Valide (Métropole)', 'Hors Métropole / DROM', 'Zéros (0,0)', 'Invalides / NaN'],
                    values=[gps_stats['valid_metro'], gps_stats['outside'], gps_stats['zeros'], gps_stats['nulls']],
                    hole=.4, marker_colors=["#198754", "#ffc107", "#dc3545", "#6c757d"]
                )])
                fig_gps.update_layout(title="Statut des coordonnées (lat/long)", height=350, margin=dict(t=50, b=0, l=0, r=0))
                st.plotly_chart(fig_gps, use_container_width=True)

                if gps_stats['comma_lat'] > 0:
                    st.warning(f"⚠️ **{gps_stats['comma_lat']:,}** lignes ont des latitudes avec des virgules. <br>👉 *Action Silver : `str.replace(',', '.')`*", unsafe_allow_html=True)

    # --- SENTINELLES -1 ---
    with c2:
        st.subheader("🎭 Détection des Sentinelles (`-1`)")
        st.markdown("Le jeu de données utilise `-1` pour encoder les valeurs manquantes. Il faut les transformer en `NaN` pour les modèles.")

        all_sentinels = []
        for table_name, df in raw_data.items():
            if not df.empty:
                s = analyze_sentinels(df)
                for col, count in s.items():
                    all_sentinels.append({"Table": table_name, "Colonne": col, "Count": count})

        if all_sentinels:
            sent_df = pd.DataFrame(all_sentinels).sort_values("Count", ascending=False).head(15)
            fig_sent = px.bar(sent_df, x="Count", y="Colonne", color="Table", orientation='h',
                              title="Top 15 des colonnes les plus impactées par '-1'")
            fig_sent.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.success("Aucune sentinelle '-1' détectée.")

    st.markdown("---")

    # --- DATES ---
    st.subheader("📅 Parsing Temporel")
    if not raw_data['caract'].empty:
        st.markdown("Fusion des colonnes `jour`, `mois`, `an`, `hrmn` en un objet `datetime` unique.")
        try:
            # Simulation du parsing
            date_str = raw_data['caract']['an'].astype(str) + "-" + raw_data['caract']['mois'].astype(str) + "-" + raw_data['caract']['jour'].astype(str)
            dates = pd.to_datetime(date_str, errors='coerce')
            invalid_dates = dates.isna().sum()

            m1, m2 = st.columns(2)
            m1.metric("Dates invalides/impossibles", f"{invalid_dates:,}")
            m2.metric("Taux de conversion", f"{(1 - invalid_dates/len(raw_data['caract']))*100:.2f} %")

            if invalid_dates > 0:
                st.warning(f"⚠️ **{invalid_dates:,}** dates ne peuvent pas être parsées (ex: jour=31 dans un mois de 30 jours). Elles deviendront `NaT` en Silver.")
        except Exception:
            st.error("Impossible d'analyser les dates (colonnes manquantes).")

# ----------------------------------------------------------------------
# ONGLET 3 : INTÉGRITÉ STRUCTURELLE
# ----------------------------------------------------------------------
with tab3:
    st.header("🔗 Intégrité Structurelle & Référentielle")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔁 Le problème de la table `lieux`")
        st.markdown("Un accident (`Num_Acc`) peut se dérouler à la jonction de plusieurs routes ou concerner plusieurs tronçons. La table `lieux` contient donc **plusieurs lignes par accident**.")

        if not raw_data['lieux'].empty:
            struct = analyze_lieux_structure(raw_data['lieux'])
            if struct:
                st.metric("Accidents avec >1 ligne", f"{struct['multi_rows']:,}")
                st.metric("Lignes 'doublons' totales", f"{struct['total_extra_rows']:,}")
                st.metric("Max lignes pour 1 accident", f"{struct['max_rows']:,}")

                if struct['multi_rows'] > 0:
                    st.error(f"🚨 **Risque d'explosion combinatoire** : Si vous joignez `lieux` sans déduplication, vous multiplierez artificiellement les accidents. <br>👉 *Action Silver : `drop_duplicates(subset=['Num_Acc'], keep='first')`*", unsafe_allow_html=True)

                # Distribution
                counts = raw_data['lieux']['Num_Acc'].value_counts()
                fig_hist = px.histogram(counts, nbins=20, title="Distribution du nombre de lignes par accident", labels={"value": "Nombre d'accidents", "count": "Lignes dans lieux"})
                st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("🔑 Intégrité Référentielle (Orphelins)")
        st.markdown("Vérification que chaque ligne des tables satellites (`lieux`, `usagers`, `vehicules`) pointe bien vers un accident existant dans `caract`.")

        if not raw_data['caract'].empty:
            valid_ids = set(raw_data['caract']['Num_Acc'])
            orphans_data = {}
            for t_name in ['lieux', 'usagers', 'vehicules']:
                if not raw_data[t_name].empty:
                    orphans = len(set(raw_data[t_name]['Num_Acc']) - valid_ids)
                    orphans_data[t_name] = orphans

            if orphans_data:
                fig_orphan = px.bar(x=list(orphans_data.keys()), y=list(orphans_data.values()),
                                    title="Nombre de lignes orphelines par table",
                                    labels={"x": "Table", "y": "Lignes orphelines"})
                fig_orphan.update_layout(showlegend=False)
                st.plotly_chart(fig_orphan, use_container_width=True)

                if sum(orphans_data.values()) == 0:
                    st.success("✅ **Parfait** : Aucune clé étrangère orpheline. Le référentiel est sain.")
                else:
                    st.warning("⚠️ Des enregistrements orphelins existent. Ils seront perdus lors d'une jointure interne (INNER JOIN).")

# ----------------------------------------------------------------------
# ONGLET 4 : LOGIQUE MÉTIER
# ----------------------------------------------------------------------
with tab4:
    st.header("🧠 Logique Métier & Enrichissement")

    if not raw_data['usagers'].empty and 'an_nais' in raw_data['usagers'].columns:
        st.subheader("🎂 Cohérence Démographique (Calcul de l'Âge)")
        st.markdown("L'âge n'est pas fourni directement. Il doit être calculé (`2024 - an_nais`). Il est crucial de détecter les aberrations avant le Feature Engineering.")

        age_stats = analyze_age_logic(raw_data['usagers'])
        if age_stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Âges Valides", f"{age_stats['valides']:,}")
            c2.metric("Âges Négatifs", f"{age_stats['negatifs']:,}", delta="Erreur" if age_stats['negatifs']>0 else None, delta_color="inverse")
            c3.metric("Centenaires (>105)", f"{age_stats['centenaires']:,}", delta="Outlier" if age_stats['centenaires']>0 else None, delta_color="inverse")
            c4.metric("Années manquantes", f"{age_stats['manquants']:,}")

            if age_stats['negatifs'] > 0 or age_stats['centenaires'] > 0:
                st.warning("⚠️ **Anomalies détectées** : Des années de naissance sont incohérentes (ex: `an_nais` = 2025 ou `an_nais` = -1). <br>👉 *Action Gold : Imputation par la médiane ou création d'une catégorie 'Âge inconnu'.*", unsafe_allow_html=True)
    else:
        st.info("💡 Le fichier `usagers-2024.csv` est vide ou ne contient pas la colonne `an_nais`. L'audit démographique est ignoré.")

    st.markdown("---")
    st.subheader("📊 Distribution des Catégories (Vérification des Dictionnaires)")
    st.markdown("Les dictionnaires de mappage (Luminosité, Météo, Type de route) couvrent-ils toutes les valeurs présentes dans la Bronze ?")

    if not raw_data['caract'].empty:
        cols_cat = st.columns(3)
        mapping = {
            "lum": {1: "Plein jour", 2: "Crépuscule", 3: "Nuit", 4: "Nuit (éclairage éteint)", 5: "Nuit (éclairage allumé)"},
            "atm": {1: "Normale", 2: "Pluie", 3: "Pluie forte", 4: "Neige", 5: "Brouillard", 6: "Vent", 7: "Temps éblouissant", 8: "Couvert", 9: "Autre"},
            "catr": {1: "Autoroute", 2: "Route Nationale", 3: "Route Départementale", 4: "Voie Communale", 5: "Hors réseau", 6: "Parking", 9: "Autre"}
        }

        for idx, (col, dico) in enumerate(mapping.items()):
            with cols_cat[idx]:
                if col in raw_data['caract'].columns:
                    vals = raw_data['caract'][col].unique()
                    unknown = [v for v in vals if v not in dico and not pd.isna(v) and v != -1]
                    if unknown:
                        st.error(f"`{col}` : Codes inconnus détectés ! {unknown}")
                    else:
                        st.success(f"`{col}` : Tous les codes sont mappés.")

# ----------------------------------------------------------------------
# ONGLET 5 : PIPELINE SILVER / GOLD
# ----------------------------------------------------------------------
with tab5:
    st.header("🚀 Plan de Transformation (Silver & Gold)")
    st.markdown("""
    <div class="alert-box alert-info">
    Basé sur l'audit ci-dessus, voici le pipeline de nettoyage et d'enrichissement qui sera appliqué pour passer de la couche <b>Bronze</b> à la couche <b>Gold</b> (prête pour le Machine Learning).
    </div>
    """, unsafe_allow_html=True)

    pipeline_data = {
        "Étape": [
            "1. Standardization", "2. Standardization", "3. Standardization",
            "4. Cleaning", "5. Cleaning", "6. Enrichment", "7. Enrichment"
        ],
        "Table Cible": [
            "caract", "caract", "lieux, usagers, vehicules",
            "lieux", "vehicules", "caract", "usagers"
        ],
        "Action Requise (Basée sur l'audit)": [
            "Créer une colonne `datetime` unifiée à partir de (jour, mois, an, hrmn).",
            "Fixer les coordonnées GPS : remplacer les virgules par des points et forcer le type numérique.",
            "Remplacer toutes les valeurs sentinelles `-1` par de vrais `NaN` (nulls).",
            "Déduplication stricte : garder une seule ligne par `Num_Acc` pour éviter l'explosion combinatoire.",
            "Nettoyage des types de véhicules et suppression des doublons.",
            "Création de features temporelles (time_of_day, weekend, etc.).",
            "Calcul de l'âge (2024 - an_nais) et gestion des outliers (âges négatifs/centenaires)."
        ],
        "Statut Audit": [
            "✅ Nécessaire" if not raw_data['caract'].empty else "⏳ En attente",
            "✅ Nécessaire" if not raw_data['caract'].empty else "⏳ En attente",
            "✅ Nécessaire" if any(not df.empty for df in raw_data.values()) else "⏳ En attente",
            "🚨 Critique" if not raw_data['lieux'].empty and analyze_lieux_structure(raw_data['lieux'])['multi_rows'] > 0 else "✅ OK",
            "✅ Nécessaire" if not raw_data['vehicules'].empty else "⏳ En attente",
            "✅ Prêt", "✅ Prêt" if not raw_data['usagers'].empty else "⏳ En attente"
        ]
    }

    df_pipeline = pd.DataFrame(pipeline_data)

    # Affichage stylisé
    st.dataframe(
        df_pipeline,
        column_config={
            "Étape": st.column_config.TextColumn("Étape", width="small"),
            "Table Cible": st.column_config.TextColumn("Table Cible", width="medium"),
            "Action Requise (Basée sur l'audit)": st.column_config.TextColumn("Action Requise (Basée sur l'audit)", width="large"),
            "Statut Audit": st.column_config.TextColumn("Statut", width="small")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### 🏆 Résultat attendu en couche Gold")
    st.markdown("""
    - **Table de Faits (`fact_accidents`)** : Une ligne unique par accident, avec des features propres (datetime, météo, géolocalisation valide, sévérité max).
    - **Tables de Dimensions** : Séparation claire des usagers et véhicules pour des analyses granulaires.
    - **Zéro Sentinelle** : Plus de `-1`, les modèles de Machine Learning (XGBoost, Random Forest) pourront gérer les `NaN` natifs ou les imputer correctement.
    """)