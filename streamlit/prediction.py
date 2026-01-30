import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from datetime import datetime
from pathlib import Path
import pandas as pd

# =========================
# CONFIGURATION (ROBUST PATHS)
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
INBOX_DIR = ROOT_DIR / "data" / "inbox"

CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# OOD / Out-of-context handling
OUT_LABEL = "Out of context"
CONF_THRESHOLD = 0.70       # tune: 0.60–0.85
MARGIN_THRESHOLD = 0.15     # tune: 0.10–0.25

# Icons + bin mapping
CLASS_UI = {
    "Cardboard": {"icon": "📦", "bin": "Bac Papier / Carton", "hint": "Sec, aplati si possible."},
    "Glass":     {"icon": "🍾", "bin": "Bac Verre / Point de collecte", "hint": "Rincer si possible, sans bouchon."},
    "Metal":     {"icon": "🥫", "bin": "Bac Recyclage (Métaux)", "hint": "Canettes, boîtes : vider puis recycler."},
    "Paper":     {"icon": "📄", "bin": "Bac Papier", "hint": "Éviter si très souillé."},
    "Plastic":   {"icon": "🧴", "bin": "Bac Plastique", "hint": "Selon règles locales (type de plastique)."},
    "Trash":     {"icon": "🗑️", "bin": "Poubelle Classique", "hint": "Non recyclable / trop sale."},
    OUT_LABEL:   {"icon": "❓", "bin": "Inconnu", "hint": "Image hors-contexte détectée."},
}

# Create Inbox folders
INBOX_DIR.mkdir(parents=True, exist_ok=True)
for c in CLASSES:
    (INBOX_DIR / c).mkdir(parents=True, exist_ok=True)

# =========================
# UI CSS (aligned with your dark eco theme)
# =========================
def _inject_prediction_css():
    st.markdown(
        """
        <style>
        .pred-hero {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            backdrop-filter: blur(6px);
            margin-bottom: 14px;
        }
        .pred-title {
            font-size: 1.6rem;
            font-weight: 950;
            letter-spacing: -0.02em;
            margin: 0 0 4px 0;
        }
        .pred-sub {
            color: rgba(230,237,243,0.74);
            margin: 0;
            font-size: 0.98rem;
        }

        .pred-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            backdrop-filter: blur(6px);
        }

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
        .pill.ood  { border-color: rgba(115,190,255,0.45); background: rgba(33,150,243,0.14); color: rgba(200,230,255,0.95); }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 16px;
            border: 1px dashed rgba(255,255,255,0.22) !important;
            background: rgba(255,255,255,0.03) !important;
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 14px;
            padding: 0.70rem 1.0rem;
            font-weight: 900;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
            color: #e6edf3;
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(46,125,50,0.45);
            background: rgba(46,125,50,0.15);
        }

        /* Compact progress look */
        .stProgress > div > div {
            border-radius: 999px;
        }

        /* Dataframe border */
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Cache model
# =========================
@st.cache_resource
def load_model():
    try:
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, len(CLASSES)))

        if MODEL_PATH.exists():
            state_dict = torch.load(str(MODEL_PATH), map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            model.eval()
            return model
        return None
    except Exception as e:
        st.error(f"Erreur modèle : {e}")
        return None


def _preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def _reset_scan():
    # Keys used in this page
    for k in ["pred_source", "pred_uploader", "pred_last_pred", "pred_last_conf"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


def show_prediction_page():
    _inject_prediction_css()

    # ---------- Header block + Reset ----------
    h1, h2 = st.columns([0.78, 0.22])
    with h1:
        st.markdown(
            """
            <div class="pred-hero">
              <div class="pred-title">📸 Scanner Intelligent</div>
              <div class="pred-sub">
                Importez une image ou utilisez la caméra. L’IA prédit la classe ou détecte un <b>hors-contexte</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h2:
        st.write("")
        st.write("")
        if st.button("🔄 Reset / New scan", use_container_width=True):
            _reset_scan()

    model = load_model()
    if model is None:
        st.error(f"⚠️ Modèle introuvable : {MODEL_PATH}")
        st.info("Vérifiez le chemin, le nom du fichier et relancez l’app.")
        return

    # ---------- Layout: Left input / Right result ----------
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="pred-card">', unsafe_allow_html=True)
        st.markdown("#### 🔎 Source de l’image")

        option = st.radio(
            "Source :",
            ["Importer fichier", "Caméra"],
            horizontal=True,
            key="pred_source",
        )

        img_file = None
        if option == "Importer fichier":
            img_file = st.file_uploader(
                "Image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="pred_uploader",
            )
        else:
            img_file = st.camera_input("Prendre une photo", key="pred_camera")

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        if not img_file:
            st.markdown(
                """
                <span class="pill">💡 Bonne lumière</span>
                <span class="pill">🎯 Objet centré</span>
                <span class="pill">🖼️ Fond simple</span>
                """,
                unsafe_allow_html=True,
            )
            st.info("Ajoutez une image pour lancer la prédiction.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        image = Image.open(img_file).convert("RGB")
        st.markdown("#### 🖼️ Aperçu")
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, width=260)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Predict ----------
    with st.spinner("Analyse en cours..."):
        x = _preprocess(image)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

    # Top-2 for margin
    top2_vals, top2_idx = torch.topk(probs, k=2)
    top1 = float(top2_vals[0].item())
    top2 = float(top2_vals[1].item())
    margin = top1 - top2

    pred = CLASSES[int(top2_idx[0].item())]
    confidence = top1 * 100

    # OOD rule
    is_ood = (top1 < CONF_THRESHOLD) or (margin < MARGIN_THRESHOLD)
    if is_ood:
        pred = OUT_LABEL

    # Save last
    st.session_state["pred_last_pred"] = pred
    st.session_state["pred_last_conf"] = confidence

   # ---------- Right panel: BIG Result + confidence + mapping + feedback ----------
    with right:
    # BIG RESULT HERO
        st.markdown(     """
        <style>
        .result-hero-box{
            background: linear-gradient(135deg, rgba(46,125,50,0.25), rgba(33,150,243,0.20));
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 22px 20px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.45);
            margin-bottom: 18px;
        }
        .result-class-big{
            font-size: 2.6rem;
            font-weight: 1000;
            letter-spacing: -0.03em;
            margin: 8px 0 6px 0;
        }
        .result-confidence-big{
            font-size: 2.1rem;
            font-weight: 900;
            opacity: 0.95;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ui = CLASS_UI.get(pred, {"icon": "❓", "bin": "Inconnu", "hint": ""})
    icon = ui["icon"]
    bin_name = ui["bin"]
    hint = ui["hint"]

    st.markdown('<div class="result-hero-box">', unsafe_allow_html=True)

    # Status pill
    if pred == OUT_LABEL:
        st.markdown('<span class="pill ood">❓ Hors contexte</span>', unsafe_allow_html=True)
    else:
        if confidence >= 80:
            st.markdown('<span class="pill good">✅ Confiant</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">⚠️ Incertain</span>', unsafe_allow_html=True)

    # BIG CLASS NAME
    if pred == OUT_LABEL:
        st.markdown(
            '<div class="result-class-big">Image hors contexte</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-class-big">{icon} {pred}</div>',
            unsafe_allow_html=True,
        )

    # BIG CONFIDENCE
    st.markdown(
        f'<div class="result-confidence-big">{confidence:.1f}% de confiance</div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.progress(min(max(confidence / 100.0, 0.0), 1.0))

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Bin mapping (secondary) ----------
    st.markdown('<div class="pred-card">', unsafe_allow_html=True)
    st.markdown("#### ♻️ Où le jeter ?")

    if pred == OUT_LABEL:
        st.info("Impossible de recommander un bac : essayez une photo plus nette et centrée.")
        st.caption(f"Confiance max: {confidence:.1f}% • Marge top1-top2: {(margin*100):.1f}%")
    else:
        st.markdown(f"**Bac recommandé :** {bin_name}")
        if hint:
            st.caption(hint)


        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        # Feedback only if not OOD
        st.markdown("#### 🧠 Feedback")
        if pred == OUT_LABEL:
            st.info("Feedback désactivé pour les images hors-contexte.")
        else:
            st.caption("Le résultat est-il correct ?")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Oui"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = INBOX_DIR / pred / f"feedback_{ts}.jpg"
                    image.save(str(save_path))
                    st.toast("Image validée ! Merci.", icon="🎉")

            with c2:
                correction = st.selectbox("Non, c'est :", CLASSES, index=CLASSES.index(pred))
                if st.button("👎 Corriger"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = INBOX_DIR / correction / f"correction_{ts}.jpg"
                    image.save(str(save_path))
                    st.toast("Correction enregistrée.", icon="📝")

        st.markdown("</div>", unsafe_allow_html=True)

        # Debug table (collapsible)
        with st.expander("📊 Détails (Top-3)"):
            topk_vals, topk_idx = torch.topk(probs, k=min(3, len(CLASSES)))
            df = pd.DataFrame(
                {
                    "Classe": [CLASSES[int(i)] for i in topk_idx],
                    "Confiance (%)": [float(v * 100) for v in topk_vals],
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                f"Seuil confiance: {int(CONF_THRESHOLD*100)}% • Seuil marge: {int(MARGIN_THRESHOLD*100)}%"
            )
