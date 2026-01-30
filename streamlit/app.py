import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="EcoLens • Tri IA",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Import pages
# -------------------------
from prediction import show_prediction_page
from admin import show_admin_page

# -------------------------
# Session navigation state
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# -------------------------
# THEME + NAVBAR (Option A: clean list with icons, hover, active accent)
# -------------------------
st.markdown(
    """
    <style>
    /* Base */
    .stApp {
        background: radial-gradient(1200px circle at 10% 10%, rgba(46,125,50,0.22), transparent 45%),
                    radial-gradient(900px circle at 90% 20%, rgba(33,150,243,0.18), transparent 45%),
                    #0b0f14;
        color: #e6edf3;
    }
    .block-container { padding-top: 2.1rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1723 0%, #0b0f14 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Text */
    h1, h2, h3, h4 { color: #e6edf3 !important; }
    p, li, label, span, div { color: #e6edf3; }
    .muted { color: rgba(230,237,243,0.72) !important; font-size: 0.95rem; }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        backdrop-filter: blur(6px);
    }
    .cardTitle { font-size: 1.05rem; font-weight: 900; margin-bottom: 8px; }

    /* Chips */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        color: rgba(230,237,243,0.90);
        font-size: 0.86rem;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .chip strong { font-weight: 900; }

    /* Hide Streamlit chrome */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ---- HERO TITLE (Gradient + Animation + Badge) ---- */
    @keyframes fadeUp { 0% {opacity:0; transform:translateY(10px);} 100% {opacity:1; transform:translateY(0);} }
    .hero-wrap { display:flex; align-items:center; gap:12px; margin-top:6px; margin-bottom:10px; animation: fadeUp 650ms ease-out both; }
    .hero-title {
        font-size: 3.1rem; font-weight: 980; letter-spacing: -0.03em; line-height: 1.05; margin: 0;
        background: linear-gradient(90deg, rgba(130,255,175,1) 0%, rgba(115,190,255,1) 55%, rgba(210,170,255,1) 100%);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-badge {
        display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
        border:1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06);
        color: rgba(230,237,243,0.90); font-weight: 900; font-size: 0.85rem;
        animation: fadeUp 750ms ease-out both; animation-delay: 80ms;
    }
    .hero-subtitle { font-size: 1.12rem; color: rgba(230,237,243,0.78); margin-bottom: 14px; animation: fadeUp 750ms ease-out both; animation-delay: 120ms; }

    /* =========================
       NAV (Option A) - clean list, subtle hover, active accent, icons + badges
       ========================= */
    :root{
      --nav-text: rgba(230,237,243,0.90);
      --nav-muted: rgba(230,237,243,0.62);
      --nav-hover: rgba(255,255,255,0.06);

      --nav-active-bg: rgba(46,125,50,0.18);
      --nav-active-border: rgba(46,125,50,0.55);
      --nav-accent: linear-gradient(180deg, rgba(130,255,175,1), rgba(115,190,255,1));
    }

    .nav-title{
      font-weight: 950;
      font-size: 0.90rem;
      letter-spacing: 0.08em;
      color: var(--nav-muted);
      margin: 8px 0 10px 0;
      text-transform: uppercase;
    }

    .nav-wrap { display:flex; flex-direction:column; gap: 4px; }

    /* Make each Streamlit button a "row" */
    .nav-row div.stButton > button,
    .nav-active div.stButton > button{
      width: 100%;
      text-align: left;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid transparent;
      background: transparent;
      color: var(--nav-text);
      font-weight: 850;
      transition: background 160ms ease, transform 160ms ease, border-color 160ms ease;
      position: relative;
      overflow: hidden;
    }

    /* Hover */
    .nav-row div.stButton > button:hover,
    .nav-active div.stButton > button:hover{
      background: var(--nav-hover);
      transform: translateX(1px);
    }

    /* Left chevron */
    .nav-row div.stButton > button::before,
    .nav-active div.stButton > button::before{
      content: "›";
      display: inline-block;
      margin-right: 10px;
      color: rgba(230,237,243,0.55);
      transform: translateY(-1px);
      font-size: 1.05rem;
    }

    /* Active row styling */
    .nav-active div.stButton > button{
      background: var(--nav-active-bg);
      border-color: var(--nav-active-border);
      color: rgba(230,237,243,0.98);
      box-shadow: 0 10px 22px rgba(0,0,0,0.35);
    }

    /* Active left accent bar */
    .nav-active div.stButton > button::after{
      content:"";
      position:absolute;
      left:0; top:0;
      height: 100%;
      width: 6px;
      background: var(--nav-accent);
    }

    .nav-active div.stButton > button::before{
      color: rgba(230,237,243,0.92);
    }

    /* Small badge */
    .nav-badge{
      display:inline-flex;
      align-items:center;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 900;
      margin-left: 10px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      color: rgba(230,237,243,0.88);
    }
    .nav-badge.green{
      border-color: rgba(46,125,50,0.40);
      background: rgba(46,125,50,0.18);
      color: #cfead2;
    }

    .soft-divider { height: 1px; background: rgba(255,255,255,0.07); margin: 14px 0 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar (Option A)
# -------------------------
st.sidebar.markdown("## ♻️ EcoLens")
st.sidebar.markdown('<div class="muted">Assistant de tri intelligent</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="nav-title">Main Menu</div>', unsafe_allow_html=True)

def nav_row(target: str, label: str, icon: str, badge_html: str = ""):
    active = (st.session_state.page == target)
    wrapper = "nav-active" if active else "nav-row"

    # Button label uses icon + text (badge is rendered separately to avoid layout hacks)
    btn_text = f"{icon} {label}"

    with st.sidebar.container():
        st.markdown(f'<div class="{wrapper}">', unsafe_allow_html=True)
        if st.button(btn_text, key=f"nav_{target}"):
            st.session_state.page = target
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Badge line (simple, clean)
        if badge_html:
            st.markdown(
                f'<div style="margin-top:-34px; margin-left: 40px; pointer-events:none;">{badge_html}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

# Keep your structure
nav_row("Accueil", "Home", "🏠")
nav_row("Scanner", "Scanner", "📸")
nav_row("Admin", "Admin", "🛠️")

# Sidebar trust/status
st.sidebar.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
with st.sidebar.container(border=True):
    st.markdown("**Statut**")
    st.success("Modèle prêt ✅")
    st.caption("PyTorch • ResNet50 • v1.0")
st.sidebar.caption("Projet Big Data & IA")

# -------------------------
# HOME PAGE
# -------------------------
def show_home():
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            """
            <div class="card">
              <div class="hero-wrap">
                <div class="hero-title">EcoLens • Tri IA</div>
                <div class="hero-badge">🏷️ v1.0</div>
              </div>

              <div class="hero-subtitle">
                Scannez un déchet et découvrez instantanément le bon bac de recyclage.
              </div>

              <div class="muted" style="margin-bottom:12px;">
                Une IA simple, rapide et responsable pour améliorer le tri sélectif au quotidien.
              </div>

              <div>
                <span class="chip">⚡ <strong>Instantané</strong></span>
                <span class="chip">📸 Caméra & Upload</span>
                <span class="chip">🧠 Feedback</span>
                <span class="chip">♻️ 6 catégories</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Commencer le scan", use_container_width=True):
                st.session_state.page = "Scanner"
                st.rerun()
        with c2:
            if st.button("📊 Tableau de bord (Admin)", use_container_width=True):
                st.session_state.page = "Admin"
                st.rerun()

        st.write("")
        with st.expander("📘 Conseils pour de meilleurs résultats"):
            st.write(
                "- Utilisez une bonne lumière (évitez les ombres).\n"
                "- Cadrez l’objet au centre.\n"
                "- Évitez le flou (photo stable).\n"
                "- Si le résultat est incertain, essayez un autre angle."
            )

    with right:
        st.markdown(
            """
            <div class="card">
              <div class="cardTitle">🎯 Démo en 10 secondes</div>
              <div class="muted">
                1) Importez une image ou utilisez la caméra<br/>
                2) L’IA reconnaît le type de déchet<br/>
                3) Le bon bac vous est indiqué<br/>
                4) Vous pouvez corriger pour améliorer le modèle
              </div>

              <div class="soft-divider"></div>

              <div class="cardTitle">🛡️ Confiance & confidentialité</div>
              <div class="muted">
                🤖 <b>Modèle</b> : ResNet50 (classification d’images)<br/>
                🔐 <b>Vie privée</b> : images de feedback stockées localement<br/>
                📚 <b>Contexte</b> : projet académique Big Data & IA
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        k1, k2, k3 = st.columns(3)
        k1.metric("Classes", "6")
        k2.metric("Mode", "Scan")
        k3.metric("Feedback", "Actif")

# -------------------------
# ROUTING
# -------------------------
if st.session_state.page == "Accueil":
    show_home()
elif st.session_state.page == "Scanner":
    show_prediction_page()
elif st.session_state.page == "Admin":
    show_admin_page()
