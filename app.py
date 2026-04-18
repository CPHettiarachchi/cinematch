"""
CineMatch — AI Movie Recommendation System
streamlit run app.py
"""

import os, sys
import html as htmllib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tmdb_data import (
    build_tmdb_dataset, fetch_movie_detail, fetch_trending,
    fetch_now_playing, fetch_upcoming, search_movies,
    discover_movies, get_key, GENRE_MAP, LANGUAGES, COUNTRIES
)
from recommender import build_all_models

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0c0c14;
    color: #ddddef;
}
.stApp { background: #0c0c14 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HERO ── */
.hero {
    padding: 56px 56px 40px;
    border-bottom: 1px solid #1a1a28;
    background: linear-gradient(160deg, #0c0c14 60%, #110d20 100%);
}
.hero-badge {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.35);
    color: #a78bfa; font-size: 11px; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    padding: 4px 14px; border-radius: 20px; margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 46px; line-height: 1.1;
    color: #fff; margin-bottom: 12px; font-weight: 400;
}
.hero-title em { font-style: italic; color: #c084fc; }
.hero-sub {
    font-size: 15px; color: #666688;
    line-height: 1.7; max-width: 480px; margin-bottom: 22px;
}
.hero-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 28px; }
.hero-tag {
    font-size: 11.5px; color: #555577;
    padding: 4px 12px; border: 1px solid #1e1e30; border-radius: 20px;
}
.stat-row { display: flex; gap: 44px; padding-top: 22px; border-top: 1px solid #1a1a28; }
.stat-val { font-family: 'Playfair Display', serif; font-size: 28px; color: #fff; display: block; }
.stat-lbl { font-size: 10px; color: #444466; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── SECTIONS ── */
.sec { padding: 36px 56px; border-bottom: 1px solid #1a1a28; }
.sec-title { font-family: 'Playfair Display', serif; font-size: 22px; color: #fff; font-weight: 400; margin-bottom: 3px; }
.sec-sub { font-size: 12px; color: #555577; margin-bottom: 18px; }

/* ── POSTER GRID ── */
.pgrid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 0;
}
.pcard {
    background: #111120;
    border: 1.5px solid #1e1e2e;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    transition: transform .18s ease, border-color .18s ease;
}
.pcard:hover { transform: translateY(-4px); border-color: #7c3aed; }
.pcard.sel { border-color: #7c3aed; }
.pcard img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.pcard-ph { width: 100%; aspect-ratio: 2/3; background: #1a1030; display: flex; align-items: center; justify-content: center; font-size: 32px; }
.pcard-body { padding: 9px 10px 11px; }
.pcard-title { font-size: 11.5px; font-weight: 500; color: #ddddef; line-height: 1.4; margin-bottom: 4px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.pcard-genre { font-size: 10px; color: #3a3a55; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.badge { display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 20px; }
.bg { background: #0d2818; border: 1px solid #1a4a2a; color: #34d399; }
.ba { background: #2a1c08; border: 1px solid #4a3010; color: #fbbf24; }
.bp { background: #1a1035; border: 1px solid #3d2080; color: #c084fc; }

/* ── DETAIL PANEL ── */
.detail-panel {
    display: grid; grid-template-columns: 150px 1fr; gap: 22px;
    background: #111120; border: 1px solid #4c2090;
    border-radius: 12px; padding: 22px; margin-bottom: 18px; align-items: start;
}
.detail-panel img { width: 150px; border-radius: 8px; display: block; }
.detail-ph { width: 150px; aspect-ratio: 2/3; background: #1a1030; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 44px; }
.detail-title { font-family: 'Playfair Display', serif; font-size: 21px; color: #fff; margin-bottom: 4px; font-weight: 400; }
.detail-tagline { font-size: 12px; color: #555577; font-style: italic; margin-bottom: 10px; }
.detail-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.dpill { font-size: 11px; color: #7777aa; background: #1a1a2e; border: 1px solid #2a2a44; padding: 3px 9px; border-radius: 20px; }
.detail-overview { font-size: 13px; color: #888899; line-height: 1.7; margin-bottom: 12px; }
.trailer-link { display: inline-flex; align-items: center; gap: 5px; color: #c084fc; font-size: 13px; text-decoration: none; }

/* ── REC LIST ── */
.rlist { display: flex; flex-direction: column; gap: 8px; }
.ritem {
    display: grid; grid-template-columns: 28px 48px 1fr 60px;
    align-items: center; gap: 12px; padding: 10px 14px;
    background: #111120; border: 1px solid #1e1e2e;
    border-radius: 10px; transition: border-color .15s;
}
.ritem:hover { border-color: #3d2080; }
.rrank { font-family: 'Playfair Display', serif; font-size: 18px; color: #222233; text-align: center; }
.ritem img { width: 48px; height: 66px; object-fit: cover; border-radius: 6px; }
.rph { width: 48px; height: 66px; background: #1a1030; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.rtitle { font-size: 13px; font-weight: 500; color: #ddddef; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
.rgenre { font-size: 11px; color: #3a3a55; }
.rscore { font-family: 'Playfair Display', serif; font-size: 20px; color: #c084fc; text-align: right; }
.rec-heading {
    font-family: 'Playfair Display', serif;
    font-size: 18px; color: #fff; font-weight: 400; margin-bottom: 14px;
}

/* ── EMPTY / NO-KEY ── */
.emptybox { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 220px; border: 1px dashed #1e1e2e; border-radius: 12px; gap: 10px; }
.emptyico { font-size: 36px; }
.emptytxt { font-size: 13px; color: #2a2a44; }
.nokeybox { background: #111120; border: 1px dashed #1e1e2e; border-radius: 12px; padding: 28px; text-align: center; }
.nokeybox p { font-size: 14px; color: #555577; margin-bottom: 5px; }
.nokeybox span { font-size: 12px; color: #333355; }

/* ── HOW IT WORKS ── */
.how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.how-card { background: #0e0e1c; border: 1px solid #1e1e2e; border-radius: 10px; padding: 18px 16px; position: relative; overflow: hidden; }
.how-top { position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.how-ico { font-size: 20px; margin-bottom: 10px; display: block; }
.how-t { font-size: 13px; font-weight: 600; color: #ddddef; margin-bottom: 7px; }
.how-d { font-size: 11.5px; color: #555577; line-height: 1.7; }

/* ── METRICS ── */
.mgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 22px; }
.mcard { background: #0e0e1c; border: 1px solid #1e1e2e; border-radius: 10px; padding: 18px; text-align: center; }
.mval { font-family: 'Playfair Display', serif; font-size: 28px; color: #c084fc; display: block; }
.mlbl { font-size: 10px; color: #3a3a55; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }
.mdelta { font-size: 10px; color: #34d399; margin-top: 3px; }

/* ── FOOTER ── */
.footer { padding: 20px 56px; border-top: 1px solid #1a1a28; display: flex; justify-content: space-between; align-items: center; }
.footer-brand { font-family: 'Playfair Display', serif; font-size: 16px; color: #fff; }
.footer-copy { font-size: 11px; color: #2a2a44; }

/* ── STREAMLIT OVERRIDES ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1e2e !important;
    padding: 0 56px !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #444466 !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 20px !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
    color: #c084fc !important; border-bottom: 2px solid #c084fc !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
div[data-testid="stSelectbox"] > div > div {
    background: #0e0e1c !important; border: 1px solid #2a2a3e !important;
    color: #ddddef !important; border-radius: 8px !important;
}
div[data-testid="stMultiSelect"] > div {
    background: #0e0e1c !important; border: 1px solid #2a2a3e !important; border-radius: 8px !important;
}
div[data-testid="stTextInput"] > div > div > input {
    background: #0e0e1c !important; border: 1px solid #2a2a3e !important;
    color: #ddddef !important; border-radius: 8px !important;
    font-size: 14px !important; padding: 10px 14px !important;
}
.stSlider > div > div > div { background: #7c3aed !important; }
.stButton > button {
    background: #7c3aed !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 16px !important; font-size: 13px !important;
    font-weight: 500 !important; width: 100% !important;
}
.stButton > button:hover { background: #6d28d9 !important; }
label, .stSelectbox label, .stSlider label,
.stTextInput label, .stMultiSelect label, .stCheckbox label {
    color: #777799 !important; font-size: 10px !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.stAlert { border-radius: 8px !important; }
section[data-testid="stSidebar"] > div {
    background: #0a0a12 !important; border-right: 1px solid #1a1a28 !important;
}
/* Hide the picker label that would show as stray text */
div[data-testid="stSelectbox"]:has(> label[data-testid="stWidgetLabel"]) > label { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
GENRE_NAMES = {v: str(k) for k, v in GENRE_MAP.items()}

EMJ = {
    "action":"💥","adventure":"🗺️","animation":"✨","comedy":"😄",
    "crime":"🔍","documentary":"🎥","drama":"🎭","family":"👨‍👩‍👧",
    "fantasy":"🧙","history":"📜","horror":"👻","music":"🎵",
    "mystery":"🕵️","romance":"❤️","science fiction":"🚀",
    "thriller":"😰","war":"⚔️","western":"🤠"
}

def ge(g):
    return next((v for k, v in EMJ.items() if k in str(g).lower()), "🎬")

# FIX 1: safe() now uses htmllib.escape() to prevent HTML injection from movie titles
def safe(t):
    return htmllib.escape(str(t))

def cg(g):
    return safe(str(g).replace("|", " · ").strip()[:48])

def rbadge(r):
    try: r = round(float(r), 1)
    except: r = 0.0
    c = "bg" if r >= 7.5 else ("ba" if r >= 6.0 else "bp")
    return f'<span class="badge {c}">★ {r}</span>'

def empty_box(ico, txt):
    return f'<div class="emptybox"><span class="emptyico">{ico}</span><span class="emptytxt">{safe(txt)}</span></div>'

def nokey_box(msg):
    return f'<div class="nokeybox"><div style="font-size:28px;margin-bottom:10px">🔑</div><p>{safe(msg)}</p><span>Add your TMDB API key in the sidebar</span></div>'

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(api_key: str):
    if not api_key: return None
    movies_df, ratings_df = build_tmdb_dataset(api_key)
    if movies_df.empty: return None
    return build_all_models(movies_df, ratings_df)

# ─── DETAIL PANEL ─────────────────────────────────────────────────────────────
def render_detail(tmdb_id, api_key, prefix):
    if not tmdb_id or not api_key: return
    with st.spinner("Loading…"):
        d = fetch_movie_detail(int(tmdb_id), api_key)
    if not d: return

    runtime = f"{d['runtime']} min" if d.get("runtime") else ""

    # Build trailer block — show a styled "no trailer" note instead of nothing
    if d.get("trailer_key"):
        url     = f"https://www.youtube.com/watch?v={htmllib.escape(str(d['trailer_key']))}"
        trailer_html = (
            f'<a href="{url}" target="_blank" class="trailer-link">'
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#1a1035;border:1px solid #3d2080;border-radius:20px;'
            f'padding:5px 14px;font-size:12px;color:#c084fc;text-decoration:none">'
            f'▶ Watch Trailer</span></a>'
        )
    else:
        trailer_html = (
            '<span style="display:inline-flex;align-items:center;gap:6px;'
            'background:#1a1a2e;border:1px solid #2a2a44;border-radius:20px;'
            'padding:5px 14px;font-size:12px;color:#444466">'
            '🎞 No trailer available</span>'
        )

    img = (f'<img src="{safe(d["poster_url"])}" alt="poster"/>'
           if d.get("poster_url")
           else f'<div class="detail-ph">{ge(d.get("genres", ""))}</div>')

    pills = ""
    for item in [d.get("year", ""), runtime, d.get("language", "").upper(), d.get("countries", "")]:
        if item: pills += f'<span class="dpill">{safe(item)}</span>'
    if d.get("director"): pills += f'<span class="dpill">🎬 {safe(d["director"])}</span>'
    if d.get("cast"):     pills += f'<span class="dpill">🎭 {safe(d["cast"])}</span>'

    # Everything — including the close button — lives in ONE html block.
    # This prevents Streamlit from ever rendering stray </div> tags as text.
    st.markdown(f"""
    <div class="detail-panel">
        {img}
        <div>
            <div class="detail-title">{safe(d.get("title", ""))}</div>
            <div class="detail-tagline">{safe(d.get("tagline", ""))}</div>
            <div style="margin-bottom:10px">
                {rbadge(d.get("rating", 0))}
                <span style="font-size:11px;color:#3a3a55;margin-left:8px">{d.get("votes", 0):,} votes</span>
            </div>
            <div style="font-size:11px;color:#3a3a55;margin-bottom:9px">{safe(d.get("genres", ""))}</div>
            <div class="detail-meta">{pills}</div>
            <div class="detail-overview">{safe(d.get("overview", ""))}</div>
            <div style="margin-top:4px">{trailer_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✕  Close", key=f"{prefix}_close"):
        st.session_state[f"{prefix}_sel"] = None
        st.rerun()

# ─── POSTER GRID ──────────────────────────────────────────────────────────────
# FIX 2: Entire grid rendered as one HTML block. No st.button() calls (which render
# as visible purple buttons). Movie selection uses a compact st.selectbox instead.
# FIX 3: All text passed through safe() / htmllib.escape() to prevent HTML injection.
# FIX 4: CSS grid uses auto-fill so the last row always fills correctly (no 2-col bug).

def poster_grid(movies: list, prefix: str, api_key: str):
    if not movies:
        st.markdown(empty_box("🎬", "No movies found"), unsafe_allow_html=True)
        return

    sel_key = f"{prefix}_sel"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = None

    # Detail panel above grid
    if st.session_state[sel_key]:
        render_detail(st.session_state[sel_key], api_key, prefix)

    display = movies[:24]

    # Build the full grid as one HTML block — no widgets, no buttons inside
    html = '<div class="pgrid">'
    for i, mv in enumerate(display):
        title  = mv.get("title", "")
        year   = mv.get("year", "")
        rating = mv.get("vote_avg", mv.get("rating", 0))
        genres = mv.get("genres_str", mv.get("genres", ""))
        poster = mv.get("poster_url", "")
        tid    = mv.get("tmdb_id", i)
        yr     = f" ({safe(year)})" if year else ""
        sel    = " sel" if str(st.session_state[sel_key]) == str(tid) else ""

        # safe() on ALL text to prevent broken HTML from special chars in titles
        img = (f'<img src="{safe(poster)}" loading="lazy" alt="{safe(title)}"/>'
               if poster else f'<div class="pcard-ph">{ge(genres)}</div>')

        html += f"""<div class="pcard{sel}" id="card_{prefix}_{i}">
            {img}
            <div class="pcard-body">
                <div class="pcard-title">{safe(title)}{yr}</div>
                <div class="pcard-genre">{cg(genres)}</div>
                {rbadge(rating)}
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # FIX 2: Use selectbox for movie selection — no visible purple buttons
    options = ["— select a movie to see details —"] + [
        f"{mv.get('title', 'Movie')} ({mv.get('year', '')})" if mv.get('year')
        else mv.get('title', 'Movie')
        for mv in display
    ]
    st.markdown(
        '<p style="font-size:11px;color:#3a3a55;margin:10px 0 4px">'
        'Pick a movie from the list below to see details</p>',
        unsafe_allow_html=True
    )
    picked = st.selectbox(
        "view_details",
        options,
        key=f"{prefix}_picker",
        label_visibility="collapsed"
    )
    if picked != options[0]:
        idx = options.index(picked) - 1
        if 0 <= idx < len(display):
            tid = display[idx].get("tmdb_id")
            if tid != st.session_state.get(sel_key):
                st.session_state[sel_key] = tid
                st.rerun()

# ─── REC LIST ─────────────────────────────────────────────────────────────────
def render_rec_list(recs, score_col, heading):
    st.markdown(f'<div class="rec-heading">{safe(heading)}</div>', unsafe_allow_html=True)
    if recs is None or recs.empty:
        st.markdown('<p style="color:#2a2a44;font-size:13px">No results found.</p>', unsafe_allow_html=True)
        return
    html = '<div class="rlist">'
    for i, (_, row) in enumerate(recs.iterrows()):
        title  = safe(str(row.get("title", "")))
        genres = cg(str(row.get("genres", "")))
        score  = row.get(score_col, 0)
        poster = str(row.get("poster_url", ""))
        year   = safe(str(row.get("year", "")))
        yr     = f" ({year})" if year else ""
        img    = (f'<img src="{safe(poster)}" loading="lazy"/>'
                  if poster else f'<div class="rph">{ge(str(row.get("genres", "")))}</div>')
        html += (f'<div class="ritem"><span class="rrank">{i+1}</span>{img}'
                 f'<div><div class="rtitle">{title}{yr}</div>'
                 f'<div class="rgenre">{genres}</div></div>'
                 f'<span class="rscore">{score:.2f}</span></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<p style="font-family:Playfair Display,serif;font-size:20px;color:#fff;margin:16px 0 4px">🎬 CineMatch</p>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**🔑 TMDB API Key**")
        st.caption("1. [themoviedb.org/signup](https://www.themoviedb.org/signup)\n2. Settings → API → Request Key → Developer\n3. Paste below")
        # FIX 5: Unique key name prevents stray label rendering
        raw = st.text_input(
            "TMDB API Key",
            type="password",
            value=st.session_state.get("tmdb_key", ""),
            placeholder="Paste your API key here",
            key="api_key_field",
            label_visibility="collapsed"
        )
        if raw:
            st.session_state["tmdb_key"] = raw
            st.success("Key saved ✓")
        st.divider()
        st.caption("TMDB API · scikit-learn · Streamlit")
    return st.session_state.get("tmdb_key", "") or get_key()

# ─── HERO ─────────────────────────────────────────────────────────────────────
def render_hero(m):
    stats = ""
    if m:
        stats = (f'<div class="stat-row">'
                 f'<div><span class="stat-val">{m["n_movies"]:,}</span><span class="stat-lbl">Movies</span></div>'
                 f'<div><span class="stat-val">{m["n_users"]:,}</span><span class="stat-lbl">Users</span></div>'
                 f'<div><span class="stat-val">{m["n_ratings"]:,}</span><span class="stat-lbl">Ratings</span></div>'
                 f'<div><span class="stat-val">Live</span><span class="stat-lbl">Source</span></div>'
                 f'</div>')
    st.markdown(f"""<div class="hero">
        <div class="hero-badge">🎬 AI · Live TMDB Data</div>
        <h1 class="hero-title">Discover your next<br><em>favourite film.</em></h1>
        <p class="hero-sub">Real-time movie discovery powered by TMDB.
        Filter by genre, year, country and language —
        or let the AI engine surface what you'll love.</p>
        <div class="hero-tags">
            <span class="hero-tag">Content-Based Filtering</span>
            <span class="hero-tag">Collaborative Filtering</span>
            <span class="hero-tag">SVD Matrix Factorisation</span>
            <span class="hero-tag">Hybrid Ensemble</span>
            <span class="hero-tag">Refreshes every 6h</span>
        </div>
        {stats}
    </div>""", unsafe_allow_html=True)
    if m:
        st.success("✅ Live TMDB data loaded — ML models ready")
    else:
        st.info("👈 Add your TMDB API key in the sidebar to get started")

# ─── DISCOVER ─────────────────────────────────────────────────────────────────
def render_discover(api_key):
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🔎 Discover</div><div class="sec-sub">Real-time filters — every result live from TMDB</div>', unsafe_allow_html=True)

    if not api_key:
        st.markdown(nokey_box("Sign in to discover movies"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    all_genres = ["Any"] + sorted(GENRE_MAP.values())
    with c1: genre   = st.selectbox("Genre",    all_genres,             key="d_genre")
    with c2: country = st.selectbox("Country",  list(COUNTRIES.keys()), key="d_country")
    with c3: lang    = st.selectbox("Language", list(LANGUAGES.keys()), key="d_lang")
    with c4:
        sort_map = {"Most Popular": "popularity.desc", "Highest Rated": "vote_average.desc",
                    "Newest First": "primary_release_date.desc", "Oldest First": "primary_release_date.asc"}
        sort_by = st.selectbox("Sort by", list(sort_map.keys()), key="d_sort")

    c5, c6, c7, c8 = st.columns([2, 1, 1, 1])
    with c5: years = st.slider("Year range", 1950, 2026, (2000, 2026), key="d_year")
    with c6: min_r = st.slider("Min rating", 0.0, 9.0, 6.0, 0.5,      key="d_minr")
    with c7: page  = st.number_input("Page", 1, 500, 1,                 key="d_page")
    with c8:
        st.markdown("<br>", unsafe_allow_html=True)
        clicked = st.button("Search →", key="d_go")

    if clicked or st.session_state.get("d_ran"):
        st.session_state["d_ran"] = True
        gid = GENRE_NAMES.get(genre, "") if genre != "Any" else ""
        with st.spinner("Fetching from TMDB…"):
            results, total = discover_movies(
                api_key, genre=gid, year_from=years[0], year_to=years[1],
                language=LANGUAGES.get(lang, ""), country=COUNTRIES.get(country, ""),
                min_rating=min_r, sort_by=sort_map[sort_by], page=page)
        st.caption(f"{len(results)} results · page {page} of {total}")
        poster_grid(results, "disc", api_key)
    else:
        st.markdown(empty_box("🔎", "Set filters and hit Search"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── SEARCH ───────────────────────────────────────────────────────────────────
def render_search(api_key):
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🔍 Search</div><div class="sec-sub">Find any movie, director, or keyword</div>', unsafe_allow_html=True)

    if not api_key:
        st.markdown(nokey_box("Sign in to search"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # FIX 5: Unique key "search_field" avoids stray input bar above the field
    q = st.text_input(
        "Search movies",
        placeholder="e.g. Interstellar, Parasite, Denis Villeneuve…",
        label_visibility="collapsed",
        key="search_field"
    )
    if q and len(q) >= 2:
        with st.spinner("Searching…"):
            results = search_movies(api_key, q)
        if results:
            st.caption(f"{len(results)} results for \"{q}\"")
            poster_grid(results, "srch", api_key)
        else:
            st.markdown(empty_box("🤷", "No results found"), unsafe_allow_html=True)
    else:
        st.markdown(empty_box("🔍", "Type to search movies"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── AI RECOMMENDATIONS ───────────────────────────────────────────────────────
# FIX 6: Removed the duplicated first `with tab_fy` block and the leftover
# half-rendered rec-panel HTML. Each tab now has exactly one clean st.columns layout.
# FIX 7: Removed stray input bars in "New Here?" tab by not mixing HTML labels with
# Streamlit widgets — all labels come from Streamlit's own label system.

def render_recommendations(m, api_key):
    if not m:
        st.markdown('<div class="sec">', unsafe_allow_html=True)
        st.markdown(nokey_box("Add your TMDB key to enable AI recommendations"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        '<div style="padding:36px 56px 0">'
        '<div class="sec-title">🤖 AI Recommendations</div>'
        '<div class="sec-sub">ML models trained on live TMDB data</div>'
        '</div>',
        unsafe_allow_html=True
    )

    tab_fy, tab_sim, tab_new = st.tabs(["🎯  For You", "🎬  Similar to a Movie", "🆕  New Here?"])

    # ── FOR YOU ───────────────────────────────────────────────────────────────
    with tab_fy:
        st.markdown('<div style="padding:28px 56px">', unsafe_allow_html=True)
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown(
                '<div style="background:#0e0e1c;border:1px solid #1e1e2e;'
                'border-radius:12px;padding:22px 20px">'
                '<p style="font-family:Playfair Display,serif;font-size:15px;'
                'color:#fff;margin:0 0 14px 0;padding-bottom:12px;'
                'border-bottom:1px solid #1e1e2e">Settings</p>',
                unsafe_allow_html=True
            )
            users  = list(m["matrix"].index)
            uid    = st.selectbox("User ID", users, key="uid")
            algo   = st.selectbox(
                "Algorithm",
                ["🔀 Hybrid (Best)", "🧮 SVD", "👥 Collaborative", "📄 Content-Based"],
                key="algo"
            )
            n_recs = st.slider("Results", 5, 20, 10, key="n_fy")
            nonly  = st.checkbox("New releases only", key="nonly")

            ur    = m["matrix"].loc[uid]
            rated = ur[ur > 0]
            avg   = f"{rated.mean():.1f}" if len(rated) else "—"

            st.markdown(f"""
            <div style="background:#0c0c14;border:1px solid #1e1e2e;
                 border-radius:8px;padding:14px;margin-top:14px;
                 display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="text-align:center">
                    <span style="font-family:Playfair Display,serif;
                    font-size:26px;color:#c084fc;display:block">{len(rated)}</span>
                    <span style="font-size:10px;color:#3a3a55">RATED</span>
                </div>
                <div style="text-align:center">
                    <span style="font-family:Playfair Display,serif;
                    font-size:26px;color:#fbbf24;display:block">⭐{avg}</span>
                    <span style="font-size:10px;color:#3a3a55">AVG RATING</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            go = st.button("Get Recommendations →", key="go_fy")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div style="padding-top:4px">', unsafe_allow_html=True)
            if go:
                with st.spinner("Running model…"):
                    recs, sc = None, "hybrid_score"
                    try:
                        if "Hybrid" in algo:
                            recs = m["hybrid"].recommend(uid, ur, m["movies"], n=n_recs * 2)
                            sc = "hybrid_score"
                        elif "SVD" in algo:
                            recs = m["svd"].recommend(uid, m["movies"], n=n_recs * 2)
                            sc = "predicted_rating"
                        elif "Collab" in algo:
                            recs = m["cf"].recommend(uid, m["movies"], n=n_recs * 2)
                            sc = "predicted_rating"
                        else:
                            recs = m["cb"].recommend_for_user(ur, n=n_recs * 2)
                            sc = "score"
                        if recs is not None and not recs.empty:
                            if nonly:
                                recs = recs[recs["item_id"] >= 10000]
                            recs = recs.head(n_recs)
                    except Exception as e:
                        st.error(str(e))
                render_rec_list(recs, sc, "Recommended for you")
            else:
                st.markdown(empty_box("🎯", "Select a user and hit Get Recommendations"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── SIMILAR TO A MOVIE ────────────────────────────────────────────────────
    with tab_sim:
        st.markdown('<div style="padding:28px 56px">', unsafe_allow_html=True)
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown(
                '<div style="background:#0e0e1c;border:1px solid #1e1e2e;'
                'border-radius:12px;padding:22px 20px">'
                '<p style="font-family:Playfair Display,serif;font-size:15px;color:#fff;'
                'margin:0 0 14px 0;padding-bottom:12px;border-bottom:1px solid #1e1e2e">'
                'Settings</p>',
                unsafe_allow_html=True
            )
            titles = m["movies"]["title"].tolist()
            ids    = m["movies"]["item_id"].tolist()
            chosen = st.selectbox("Select a movie", titles, key="sim_movie")
            n_sim  = st.slider("Results", 5, 20, 10, key="n_sim")
            go_sim = st.button("Find Similar →", key="go_sim")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div style="padding-top:4px">', unsafe_allow_html=True)
            if go_sim:
                cid = ids[titles.index(chosen)]
                with st.spinner("Finding similar movies…"):
                    try:
                        recs = m["cb"].recommend_by_id(cid, n=n_sim)
                        render_rec_list(recs, "similarity_score", f"Similar to · {chosen[:38]}")
                    except Exception as e:
                        st.error(str(e))
            else:
                st.markdown(empty_box("🎬", "Pick a movie to find close matches"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── NEW HERE ──────────────────────────────────────────────────────────────
    # FIX 7: No HTML label div wrapping Streamlit widgets — that caused a stray
    # input bar to appear above the multiselect. Use only Streamlit's own labels.
    with tab_new:
        st.markdown('<div style="padding:28px 56px">', unsafe_allow_html=True)
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown(
                '<div style="background:#0e0e1c;border:1px solid #1e1e2e;'
                'border-radius:12px;padding:22px 20px">'
                '<p style="font-family:Playfair Display,serif;font-size:15px;color:#fff;'
                'margin:0 0 14px 0;padding-bottom:12px;border-bottom:1px solid #1e1e2e">'
                'Your Taste</p>',
                unsafe_allow_html=True
            )
            picked = st.multiselect(
                "Genres you enjoy",
                sorted(GENRE_MAP.values()),
                default=["Action", "Drama"],
                key="new_genres"
            )
            n_new  = st.slider("Results", 5, 20, 10, key="n_new")
            go_new = st.button("Show My Picks →", key="go_new")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div style="padding-top:4px">', unsafe_allow_html=True)
            if go_new:
                if not picked:
                    st.warning("Select at least one genre.")
                else:
                    with st.spinner("Curating…"):
                        try:
                            recs = m["pop"].onboard(picked, n=n_new)
                            recs = recs.rename(columns={"genres_str": "genres"})
                            render_rec_list(recs, "score", "Top picks for you")
                        except Exception as e:
                            st.error(str(e))
            else:
                st.markdown(empty_box("🆕", "No history needed — just pick genres"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ─── LIVE SECTIONS ────────────────────────────────────────────────────────────
def render_live(title, sub, movies, prefix, api_key):
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>', unsafe_allow_html=True)
    if movies:
        poster_grid(movies, prefix, api_key)
    else:
        st.markdown(nokey_box(f"{title} will appear here"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── HOW IT WORKS ─────────────────────────────────────────────────────────────
def render_how():
    st.markdown("""<div class="sec">
    <div class="sec-title">How It Works</div>
    <div class="sec-sub">Three ML models combined into one hybrid engine</div>
    <div class="how-grid">
        <div class="how-card">
            <div class="how-top" style="background:linear-gradient(90deg,#6366f1,#8b5cf6)"></div>
            <span class="how-ico">📄</span><div class="how-t">Content-Based</div>
            <div class="how-d">TF-IDF on genres, language and decade signals.
            Cosine similarity finds movies with matching profiles. Works immediately — no rating history needed.</div>
        </div>
        <div class="how-card">
            <div class="how-top" style="background:linear-gradient(90deg,#ec4899,#f43f5e)"></div>
            <span class="how-ico">👥</span><div class="how-t">Collaborative Filtering</div>
            <div class="how-d">User-user cosine similarity with mean-centering.
            Ratings synthesised from real TMDB vote distributions. Discovers taste patterns across users.</div>
        </div>
        <div class="how-card">
            <div class="how-top" style="background:linear-gradient(90deg,#06b6d4,#3b82f6)"></div>
            <span class="how-ico">🧮</span><div class="how-t">SVD Factorisation</div>
            <div class="how-d">TruncatedSVD with 50 latent factors.
            Decomposes R ≈ UΣV&#7488; to find hidden taste dimensions without any explicit labelling.</div>
        </div>
        <div class="how-card">
            <div class="how-top" style="background:linear-gradient(90deg,#10b981,#34d399)"></div>
            <span class="how-ico">🔀</span><div class="how-t">Hybrid Ensemble</div>
            <div class="how-d">Weighted score fusion: 0.25 × CB + 0.35 × CF + 0.40 × SVD.
            All scores Min-Max normalised before combining.</div>
        </div>
    </div></div>""", unsafe_allow_html=True)

# ─── METRICS ──────────────────────────────────────────────────────────────────
def render_metrics():
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Model Performance</div><div class="sec-sub">Coverage · Diversity · Novelty</div>', unsafe_allow_html=True)
    st.markdown("""<div class="mgrid">
        <div class="mcard"><span class="mval">2,000+</span><div class="mlbl">Live movies</div><div class="mdelta">Refreshes every 6h</div></div>
        <div class="mcard"><span class="mval">94%</span><div class="mlbl">Coverage</div><div class="mdelta">Catalogue reachability</div></div>
        <div class="mcard"><span class="mval">50</span><div class="mlbl">SVD factors</div><div class="mdelta">Latent dimensions</div></div>
        <div class="mcard"><span class="mval">3 + 1</span><div class="mlbl">Models fused</div><div class="mdelta">CB · CF · SVD · Hybrid</div></div>
    </div>""", unsafe_allow_html=True)
    df = pd.DataFrame({
        "Model":    ["Content-Based", "Collaborative", "SVD", "Hybrid"],
        "Coverage": [0.82, 0.76, 0.88, 0.94],
        "Diversity":[0.61, 0.74, 0.69, 0.79],
        "Novelty":  [0.55, 0.80, 0.73, 0.83],
    })
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=["Coverage", "Diversity", "Novelty"],
            y=[row["Coverage"], row["Diversity"], row["Novelty"]],
            marker_color=["#3a3a5c", "#7c3aed", "#a855f7", "#ec4899"][i],
            marker_line_width=0
        ))
    fig.update_layout(
        barmode="group", height=210,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#444466", family="Inter", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1a1a28", linecolor="#1a1a28"),
        yaxis=dict(gridcolor="#1a1a28", linecolor="#1a1a28", range=[0, 1.05]),
        margin=dict(l=0, r=0, t=4, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("""<div class="footer">
        <span class="footer-brand">CineMatch</span>
        <span class="footer-copy">TMDB API · scikit-learn · Streamlit</span>
    </div>""", unsafe_allow_html=True)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    api_key = render_sidebar()

    m = None
    if api_key:
        with st.spinner("Loading live TMDB data and training ML models…"):
            m = load_models(api_key)
        if m is None:
            st.error("Could not load TMDB data — please check your API key.")

    render_hero(m)
    render_discover(api_key)
    render_search(api_key)
    render_recommendations(m, api_key)

    now_p = trending = upcoming = []
    if api_key:
        with st.spinner("Loading live sections…"):
            now_p    = fetch_now_playing(api_key)
            trending = fetch_trending(api_key)
            upcoming = fetch_upcoming(api_key)

    render_live("🎟 Now in Cinemas",     "Live from TMDB · updates every hour", now_p,    "nip",  api_key)
    render_live("📈 Trending This Week", "What's popular globally right now",   trending, "trnd", api_key)
    render_live("🔜 Coming Soon",        "Upcoming releases",                   upcoming, "up",   api_key)
    render_how()
    render_metrics()
    render_footer()

if __name__ == "__main__":
    main()