import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# =========================
# Robust paths (dashboard.py is in PROJET_PI/streamlit/)
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
INBOX_DIR = ROOT_DIR / "data" / "inbox"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"

CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _safe_count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    n = 0
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n += 1
    return n


def _inbox_counts(inbox_dir: Path, classes: list[str]) -> pd.DataFrame:
    rows = []
    if not inbox_dir.exists():
        for c in classes:
            rows.append({"Classe": c, "Images": 0})
        df = pd.DataFrame(rows)
        return df

    for c in classes:
        cdir = inbox_dir / c
        rows.append({"Classe": c, "Images": _safe_count_images(cdir)})

    df = pd.DataFrame(rows).sort_values("Images", ascending=False)
    return df


def _inbox_recent_activity(inbox_dir: Path) -> pd.DataFrame:
    """
    Build a small "activity" table: counts per day based on file modified time.
    If inbox missing/empty, returns empty df.
    """
    if not inbox_dir.exists():
        return pd.DataFrame(columns=["Date", "Images"])

    records = []
    for p in inbox_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            dt = datetime.fromtimestamp(p.stat().st_mtime).date()
            records.append(dt)

    if not records:
        return pd.DataFrame(columns=["Date", "Images"])

    s = pd.Series(records, name="Date").value_counts().sort_index()
    df = s.rename_axis("Date").reset_index(name="Images")
    return df


def _model_info(model_path: Path) -> dict:
    if not model_path.exists():
        return {"exists": False}

    stt = model_path.stat()
    mtime = datetime.fromtimestamp(stt.st_mtime).strftime("%Y-%m-%d %H:%M")
    size_mb = stt.st_size / (1024 * 1024)
    return {"exists": True, "mtime": mtime, "size_mb": size_mb}


def _retrain_readiness(total_inbox: int) -> tuple[str, str]:
    """
    Simple, honest recommendation.
    You can tune thresholds for your project.
    """
    if total_inbox >= 200:
        return "Prêt ✅", "Vous avez assez de feedback pour relancer un entraînement significatif."
    if total_inbox >= 80:
        return "Bientôt ✅", "Encore un peu de feedback pour stabiliser la distribution des classes."
    if total_inbox >= 20:
        return "En collecte 🟡", "Continuez à collecter du feedback (objectif: ~80+ images)."
    return "Insuffisant 🔴", "Collectez plus d'images via le feedback utilisateur (objectif: 20+ puis 80+)."


def show_dashboard():
    st.title("📊 Admin • Monitoring & Feedback (EcoLens)")

    # =========================
    # KPIs
    # =========================
    inbox_df = _inbox_counts(INBOX_DIR, CLASSES)
    total_inbox = int(inbox_df["Images"].sum())
    classes_nonzero = int((inbox_df["Images"] > 0).sum())

    model = _model_info(MODEL_PATH)
    model_status = "Chargé ✅" if model.get("exists") else "Introuvable ❌"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inbox (feedback)", f"{total_inbox}")
    col2.metric("Classes couvertes", f"{classes_nonzero}/{len(CLASSES)}")
    col3.metric("Modèle", model_status)
    col4.metric("Version", "v1.0 • ResNet50")

    # Model details (honest)
    with st.expander("ℹ️ Détails du modèle"):
        if model.get("exists"):
            st.write(f"**Fichier** : `{MODEL_PATH}`")
            st.write(f"**Dernière mise à jour** : {model['mtime']}")
            st.write(f"**Taille** : {model['size_mb']:.2f} MB")
            st.caption("Le dashboard se base sur des métriques live (feedback). Les évaluations offline restent dans Colab/Kaggle.")
        else:
            st.warning(f"Le fichier modèle est introuvable : `{MODEL_PATH}`")
            st.caption("Vérifiez que `models/best_model.pth` existe.")

    st.divider()

    # =========================
    # Feedback Analytics (Inbox)
    # =========================
    st.subheader("📁 Feedback utilisateur — Distribution par classe")
    if total_inbox == 0:
        st.info("Aucune image dans `data/inbox/` pour le moment. Faites quelques feedbacks depuis la page Scanner.")
    else:
        st.bar_chart(inbox_df.set_index("Classe")["Images"])
        st.caption("Ces données proviennent d’images validées/corrigées par les utilisateurs (production-like).")

        # Imbalance insight
        max_c = inbox_df.iloc[0]
        min_c = inbox_df.iloc[-1]
        if min_c["Images"] == 0 and max_c["Images"] > 0:
            st.warning(
                f"Déséquilibre détecté : **{max_c['Classe']}** domine, "
                f"et au moins une classe est à **0**. Collectez du feedback sur les classes manquantes."
            )
        elif min_c["Images"] > 0:
            ratio = float(max_c["Images"] / max(1, min_c["Images"]))
            if ratio >= 3:
                st.warning(
                    f"Déséquilibre : la classe **{max_c['Classe']}** a ~{ratio:.1f}× plus d’images que **{min_c['Classe']}**. "
                    "Un ré-équilibrage améliorera la robustesse."
                )
            else:
                st.success("Distribution relativement équilibrée ✅")

    st.divider()

    # =========================
    # Activity over time
    # =========================
    st.subheader("📈 Activité de collecte (Inbox) — dans le temps")
    activity_df = _inbox_recent_activity(INBOX_DIR)
    if activity_df.empty:
        st.info("Pas encore assez d’historique de feedback pour afficher une courbe.")
    else:
        st.line_chart(activity_df.set_index("Date")["Images"])
        st.caption("Basé sur la date de modification des images dans `data/inbox/`.")

    st.divider()

    # =========================
    # MLOps-lite: Retraining readiness
    # =========================
    st.subheader("🧪 MLOps simplifié — Readiness de ré-entraînement")

    status, recommendation = _retrain_readiness(total_inbox)
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        st.markdown("**Statut**")
        st.markdown(f"### {status}")
    with c2:
        st.markdown("**Recommandation**")
        st.write(recommendation)

    st.markdown(
        """
        **Règle simple (projet académique) :**
        - < 20 images → trop tôt  
        - 20–80 → collecter davantage  
        - 80–200 → bon moment pour tester un ré-entraînement  
        - > 200 → très bon moment (meilleure généralisation)
        """
    )

    st.divider()

    # =========================
    # Model Behavior note (since no confusion matrix file)
    # =========================
    st.subheader("🤖 Comportement du modèle (inférence)")
    st.info(
        "Dans l'app, chaque prédiction est accompagnée d’un **score de confiance** "
        "et d’une **détection hors-contexte (OOD)** (rejet si confiance faible ou ambiguïté)."
    )
    st.markdown(
        """
        **Pourquoi pas de matrice de confusion ici ?**  
        - Les évaluations classiques (accuracy / confusion matrix) sont réalisées **offline** (Colab/Kaggle).  
        - Le dashboard Admin se concentre sur la **surveillance live** : feedback utilisateur, collecte, équilibre des classes, readiness retraining.
        """
    )

    # Optional session stats if you store them elsewhere
    if "stats" in st.session_state and isinstance(st.session_state["stats"], dict):
        with st.expander("📌 Stats session (optionnel)"):
            st.json(st.session_state["stats"])
