import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io
import hashlib

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================================
# Paths
# admin.py in PROJET_PI/streamlit/
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT_DIR / "Garbage classification"

CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================================
# Optional auth (keep if you already have auth.py)
# ==========================================================
def _require_auth() -> bool:
    try:
        from auth import check_password
        return bool(check_password())
    except Exception:
        return True


# ==========================================================
# Theme CSS (dark eco cards)
# ==========================================================
def _inject_css():
    st.markdown(
        """
        <style>
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            backdrop-filter: blur(6px);
            margin-bottom: 14px;
        }
        .cardTitle {
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .muted { color: rgba(230,237,243,0.72) !important; font-size: 0.95rem; }
        .soft-divider {
            height: 1px;
            background: rgba(255,255,255,0.07);
            margin: 14px 0 12px 0;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
            color: rgba(230,237,243,0.90);
            font-size: 0.86rem;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .pill.good { border-color: rgba(46,125,50,0.45); background: rgba(46,125,50,0.18); color: #cfead2; }
        .pill.warn { border-color: rgba(255,193,7,0.45); background: rgba(255,193,7,0.12); color: rgba(255,233,180,0.95); }
        .pill.info { border-color: rgba(115,190,255,0.45); background: rgba(33,150,243,0.14); color: rgba(200,230,255,0.95); }
        .pill.bad  { border-color: rgba(255,90,90,0.45); background: rgba(255,90,90,0.10); color: rgba(255,210,210,0.95); }

        .scoreWrap{
            display:flex; align-items:center; justify-content:space-between;
            gap:14px;
            padding: 12px 12px;
            border-radius: 16px;
            background: rgba(0,0,0,0.18);
            border: 1px solid rgba(255,255,255,0.08);
            margin-top: 10px;
        }
        .scoreNum{
            font-size: 2.1rem;
            font-weight: 1000;
            letter-spacing: -0.03em;
            margin: 0;
        }
        .scoreLabel{
            font-size: 0.95rem;
            color: rgba(230,237,243,0.72);
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Dataset discovery helpers
# ==========================================================
def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _detect_splits(dataset_root: Path):
    """
    Supports:
    - dataset_root/<class>/
    - dataset_root/train/<class>/, val/<class>/, test/<class>/
    """
    if not dataset_root.exists():
        return []

    split_candidates = ["train", "val", "valid", "validation", "test"]
    found = []
    for s in split_candidates:
        d = dataset_root / s
        if d.exists() and d.is_dir():
            found.append(s)

    return found if found else ["(root)"]


def _get_split_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root if split == "(root)" else dataset_root / split


def _count_class_images(split_dir: Path) -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        cdir = split_dir / c
        n = 0
        if cdir.exists():
            n = sum(1 for p in cdir.rglob("*") if _is_image(p))
        rows.append({"Classe": c, "Images": n})
    df = pd.DataFrame(rows).sort_values("Images", ascending=False)
    total = max(1, int(df["Images"].sum()))
    df["Part (%)"] = (df["Images"] / total * 100).round(2)
    return df


def _iter_images(split_dir: Path):
    for c in CLASSES:
        cdir = split_dir / c
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if _is_image(p):
                yield c, p


def _quick_hash(p: Path, block_size=65536) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _scan_dataset_stats(split_dir_str: str, sample_size: int):
    """
    Sample-based scan:
    - corrupted images
    - width/height, aspect
    - brightness (grayscale mean)
    - file size
    """
    split_dir = Path(split_dir_str)

    all_paths = [(c, p) for c, p in _iter_images(split_dir)]
    if not all_paths:
        return pd.DataFrame(), pd.DataFrame(), 0

    total = len(all_paths)
    rng = np.random.default_rng(42)

    if sample_size and sample_size < total:
        idx = rng.choice(total, size=sample_size, replace=False)
        chosen = [all_paths[i] for i in idx]
    else:
        chosen = all_paths

    records = []
    bad = []

    for c, p in chosen:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                w, h = img.size
                aspect = w / max(1, h)

                gray = img.convert("L")
                brightness = float(np.array(gray).mean())

            file_kb = p.stat().st_size / 1024.0

            records.append(
                {
                    "class": c,
                    "path": str(p),
                    "width": int(w),
                    "height": int(h),
                    "aspect": float(aspect),
                    "brightness": float(brightness),
                    "file_kb": float(file_kb),
                }
            )
        except Exception as e:
            bad.append({"class": c, "path": str(p), "error": str(e)})

    return pd.DataFrame(records), pd.DataFrame(bad), total


@st.cache_data(show_spinner=False)
def _detect_duplicates(split_dir_str: str, max_files_to_hash: int):
    """
    Duplicate detection by file hash (limited)
    """
    split_dir = Path(split_dir_str)
    all_paths = [p for _, p in _iter_images(split_dir)]
    if not all_paths:
        return pd.DataFrame()

    if max_files_to_hash and len(all_paths) > max_files_to_hash:
        all_paths = all_paths[:max_files_to_hash]

    seen = {}
    dups = []
    for p in all_paths:
        try:
            hx = _quick_hash(p)
            if hx in seen:
                dups.append({"hash": hx, "path": str(p), "duplicate_of": str(seen[hx])})
            else:
                seen[hx] = p
        except Exception:
            continue

    return pd.DataFrame(dups)


# ==========================================================
# Health Score + Actions
# ==========================================================
def _health_score_and_actions(df_counts: pd.DataFrame, df_stats: pd.DataFrame, df_bad: pd.DataFrame, df_dups: pd.DataFrame):
    """
    Returns:
      score (0..100),
      badges list,
      prioritized actions list (dicts with priority/severity/title/action)
    """
    score = 100
    actions = []
    badges = []

    total = int(df_counts["Images"].sum())
    missing_classes = df_counts[df_counts["Images"] == 0]["Classe"].tolist()

    if total == 0:
        score = 0
        actions.append({
            "severity": "bad",
            "title": "Dataset vide ou structure invalide",
            "action": "Vérifiez que le dossier contient un sous-dossier par classe (Cardboard, Glass, Metal, Paper, Plastic, Trash)."
        })
        return score, badges, actions

    # 1) Missing classes (major)
    if missing_classes:
        score -= min(45, 10 * len(missing_classes))
        actions.append({
            "severity": "bad",
            "title": f"Classes manquantes: {', '.join(missing_classes)}",
            "action": "Ajoutez des images pour ces classes ou corrigez la structure des dossiers."
        })

    # 2) Imbalance (major)
    max_row = df_counts.iloc[0]
    min_row = df_counts.iloc[-1]
    max_n = int(max_row["Images"])
    min_n = int(min_row["Images"])

    if min_n == 0 and max_n > 0:
        score -= 25
        actions.append({
            "severity": "warn",
            "title": "Déséquilibre fort (au moins une classe à 0)",
            "action": "Collectez/augmentez les classes faibles. Sinon le modèle sera biaisé."
        })
    elif min_n > 0:
        ratio = max_n / max(1, min_n)
        if ratio >= 5:
            score -= 25
            actions.append({
                "severity": "warn",
                "title": f"Déséquilibre fort: {max_row['Classe']} ≈ {ratio:.1f}× {min_row['Classe']}",
                "action": "Ajoutez des données pour les classes faibles ou utilisez un échantillonnage pondéré (WeightedSampler)."
            })
        elif ratio >= 3:
            score -= 15
            actions.append({
                "severity": "warn",
                "title": f"Déséquilibre modéré: {max_row['Classe']} ≈ {ratio:.1f}× {min_row['Classe']}",
                "action": "Améliorez l’équilibre pour augmenter la robustesse."
            })
        else:
            badges.append(("good", "Distribution OK"))

    # 3) Corrupted images
    if not df_bad.empty:
        n_bad = len(df_bad)
        score -= min(20, 1 + int(n_bad * 0.8))
        actions.append({
            "severity": "warn",
            "title": f"Fichiers illisibles/corrompus: {n_bad}",
            "action": "Supprimez/remplacez ces fichiers avant l’entraînement."
        })

    # 4) Duplicates
    if df_dups is not None and not df_dups.empty:
        n_dups = len(df_dups)
        score -= min(15, 2 + int(n_dups * 0.5))
        actions.append({
            "severity": "warn",
            "title": f"Doublons potentiels détectés: {n_dups}",
            "action": "Supprimez les doublons (ou dédupliquez) pour éviter un overfitting sur certaines images."
        })

    # 5) Image stats (sample-based)
    if df_stats is not None and not df_stats.empty:
        n = len(df_stats)

        small = df_stats[(df_stats["width"] < 128) | (df_stats["height"] < 128)]
        extreme_aspect = df_stats[(df_stats["aspect"] > 2.2) | (df_stats["aspect"] < 0.45)]
        dark = df_stats[df_stats["brightness"] < 45]
        bright = df_stats[df_stats["brightness"] > 210]

        # Penalize only if significant
        def _flag(count, label, penalty, action_text, severity="warn"):
            nonlocal score
            if count > max(3, int(0.05 * n)):
                score -= penalty
                actions.append({
                    "severity": severity,
                    "title": f"{label}: ~{count} images (échantillon)",
                    "action": action_text
                })

        _flag(len(small), "Beaucoup d’images très petites", 8,
              "Préférez des images plus nettes. Sinon le modèle apprend du bruit.")
        _flag(len(extreme_aspect), "Ratios extrêmes (cadrage/crop)", 7,
              "Vérifiez les crops. Pensez à RandomResizedCrop au training.")
        _flag(len(dark), "Images très sombres", 6,
              "Ajoutez des augmentations (ColorJitter brightness/contrast) + conseillez une meilleure lumière.")
        _flag(len(bright), "Images très lumineuses/reflets", 6,
              "Ajoutez ColorJitter/contrast et évitez les reflets pendant la capture.")

    # clamp score
    score = int(max(0, min(100, score)))

    # badges by score
    if score >= 85:
        badges.append(("good", "Excellent"))
    elif score >= 70:
        badges.append(("info", "Bon"))
    elif score >= 50:
        badges.append(("warn", "Moyen"))
    else:
        badges.append(("bad", "À corriger"))

    # prioritize actions: bad first then warn
    severity_rank = {"bad": 0, "warn": 1, "info": 2, "good": 3}
    actions = sorted(actions, key=lambda a: severity_rank.get(a["severity"], 9))

    return score, badges, actions


def _pill(sev: str) -> str:
    if sev == "good":
        return '<span class="pill good">OK</span>'
    if sev == "info":
        return '<span class="pill info">Info</span>'
    if sev == "warn":
        return '<span class="pill warn">Attention</span>'
    return '<span class="pill bad">Critique</span>'


# ==========================================================
# Admin Page
# ==========================================================
def show_admin_page():
    _inject_css()

    if not _require_auth():
        return

    st.title("🛠️ Admin • Dataset Insights")
    st.markdown(
        '<div class="muted">Exploration du dataset local <code>Garbage classification</code> + Health Score + Actions recommandées.</div>',
        unsafe_allow_html=True,
    )

    if not DATASET_ROOT.exists():
        st.error(f"Dataset introuvable: {DATASET_ROOT}")
        st.info("Créez le dossier `Garbage classification/` à la racine du projet (même niveau que `streamlit/`).")
        return

    # Compact split selector (no big "Source dataset" section)
    splits = _detect_splits(DATASET_ROOT)
    split = st.selectbox("Dataset split", splits, index=0, help="Auto-détecte train/val/test si présents.")
    split_dir = _get_split_dir(DATASET_ROOT, split)
    st.caption(f"Chemin: {split_dir}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧼 Qualité & Actions", "🖼️ Samples"])

    # ======================================================
    # TAB 1: Overview
    # ======================================================
    with tab1:
        df_counts = _count_class_images(split_dir)
        total = int(df_counts["Images"].sum())
        missing = int((df_counts["Images"] == 0).sum())
        covered = len(CLASSES) - missing

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="cardTitle">📊 Vue globale</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Images", f"{total}")
        k2.metric("Classes couvertes", f"{covered}/{len(CLASSES)}")
        k3.metric("Classes manquantes", f"{missing}")
        k4.metric("Split", split)

        st.bar_chart(df_counts.set_index("Classe")["Images"])

        st.download_button(
            "⬇️ Télécharger distribution (CSV)",
            data=df_counts.to_csv(index=False).encode("utf-8"),
            file_name=f"dataset_distribution_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # TAB 2: Quality & Actions (Health Score + prioritized actions)
    # ======================================================
    with tab2:
        df_counts = _count_class_images(split_dir)

        # Fixed: no more "Paramètres analyse" section; use sensible defaults
        SAMPLE_SIZE = 800
        DUP_LIMIT = 1200

        with st.spinner("Analyse offline du dataset (échantillon + checks)..."):
            df_stats, df_bad, total_found = _scan_dataset_stats(str(split_dir), SAMPLE_SIZE)
            df_dups = _detect_duplicates(str(split_dir), DUP_LIMIT)

        score, badges, actions = _health_score_and_actions(df_counts, df_stats, df_bad, df_dups)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="cardTitle">🩺 Dataset Health Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Score global (0–100) basé sur: équilibre, classes manquantes, corruption, doublons, qualité image.</div>', unsafe_allow_html=True)

        # Score block
        st.markdown(
            f"""
            <div class="scoreWrap">
              <div>
                <p class="scoreNum">{score}/100</p>
                <p class="scoreLabel">Qualité dataset (offline)</p>
              </div>
              <div style="text-align:right;">
                {" ".join([_pill(sev) + f" <b>{txt}</b>" for sev, txt in badges])}
                <div class="muted" style="margin-top:6px;">Échantillon stats: {min(SAMPLE_SIZE, total_found)} images • Doublons check: {DUP_LIMIT} fichiers</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        # Actions (prioritized)
        st.markdown("#### ✅ What to fix first (priorité)")
        if not actions:
            st.success("Aucun problème détecté ✅")
        else:
            for i, a in enumerate(actions[:8], start=1):
                st.markdown(
                    f"""{_pill(a["severity"])} <b>{i}. {a["title"]}</b><br>
                    <span class="muted">{a["action"]}</span>""",
                    unsafe_allow_html=True,
                )

        # Export action report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split": split,
            "dataset_root": str(DATASET_ROOT),
            "health_score": score,
            "sample_size": min(SAMPLE_SIZE, total_found),
            "dup_limit": DUP_LIMIT,
        }
        df_actions = pd.DataFrame([{"priority": i + 1, **a} for i, a in enumerate(actions)])
        df_meta = pd.DataFrame(list(report.items()), columns=["metric", "value"])

        out = io.StringIO()
        df_meta.to_csv(out, index=False)
        out.write("\n")
        df_counts.to_csv(out, index=False)
        out.write("\n")
        df_actions.to_csv(out, index=False)
        actions_csv = out.getvalue().encode("utf-8")

        st.download_button(
            "⬇️ Télécharger Health Report (CSV)",
            data=actions_csv,
            file_name=f"dataset_health_report_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Details sections (still in same tab, but secondary)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="cardTitle">📐 Détails (qualité images)</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Histogrammes calculés sur un échantillon (rapide, offline).</div>', unsafe_allow_html=True)

        if df_stats.empty:
            st.info("Aucune image trouvée pour calculer les statistiques.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Résolutions (width & height)")
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.hist(df_stats["width"], bins=25, alpha=0.8, label="width")
                ax.hist(df_stats["height"], bins=25, alpha=0.8, label="height")
                ax.set_xlabel("pixels")
                ax.set_ylabel("count")
                ax.legend()
                st.pyplot(fig)

            with c2:
                st.caption("Aspect ratio (width/height)")
                fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
                ax2.hist(df_stats["aspect"], bins=30)
                ax2.set_xlabel("aspect")
                ax2.set_ylabel("count")
                st.pyplot(fig2)

            c3, c4 = st.columns(2)
            with c3:
                st.caption("Brightness (moyenne grayscale)")
                fig3, ax3 = plt.subplots(figsize=(6.5, 4.2))
                ax3.hist(df_stats["brightness"], bins=30)
                ax3.set_xlabel("brightness (0..255)")
                ax3.set_ylabel("count")
                st.pyplot(fig3)

            with c4:
                st.caption("Taille fichier (KB)")
                fig4, ax4 = plt.subplots(figsize=(6.5, 4.2))
                ax4.hist(df_stats["file_kb"], bins=30)
                ax4.set_xlabel("KB")
                ax4.set_ylabel("count")
                st.pyplot(fig4)

        # Corrupted files
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🧯 Fichiers corrompus")
        if df_bad.empty:
            st.success("Aucun fichier corrompu détecté sur l’échantillon ✅")
        else:
            st.warning(f"{len(df_bad)} fichier(s) problématique(s) détecté(s).")
            st.dataframe(df_bad.head(200), use_container_width=True, hide_index=True)

        # Duplicates
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🧬 Doublons potentiels (hash fichier)")
        if df_dups.empty:
            st.success("Aucun doublon détecté dans la limite définie ✅")
            st.caption("Note: la détection est limitée, augmentez DUP_LIMIT si besoin.")
        else:
            st.warning(f"Doublons détectés : {len(df_dups)}")
            st.dataframe(df_dups.head(200), use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # TAB 3: Samples
    # ======================================================
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="cardTitle">🖼️ Échantillons par classe</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Contrôle visuel: cohérence labels, qualité, diversité.</div>', unsafe_allow_html=True)

        chosen_class = st.selectbox("Classe", CLASSES, index=0)
        chosen_dir = split_dir / chosen_class

        if not chosen_dir.exists():
            st.warning(f"Dossier manquant: {chosen_dir}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        imgs = [p for p in chosen_dir.rglob("*") if _is_image(p)]
        if not imgs:
            st.warning("Aucune image dans ce dossier.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        np.random.shuffle(imgs)
        imgs = imgs[:12]

        cols = st.columns(4)
        for i, p in enumerate(imgs):
            with cols[i % 4]:
                st.image(str(p), caption=p.name, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
