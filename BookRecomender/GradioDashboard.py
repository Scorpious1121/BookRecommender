import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

load_dotenv()

# ── Config ────────────────────────────────────────────────
TONE_MAP = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
TOP_K = {"initial": 50, "final": 16}

HERO = {
    "title": "The Name<br>of the Wind",
    "author": "Patrick Rothfuss",
    "genre": "Fantasy",
    "rating": "4.8",
    "pages": "662",
}

NAV_LINKS = ["Home", "Series", "New releases", "My list"]
CATEGORIES_EXTRA = ["Fantasy", "Sci-fi", "Thriller", "Mystery", "Romance", "History", "Horror"]

# ── Data & vector store ───────────────────────────────────
def load_data():
    df = pd.read_csv("books_with_emotions.csv")
    df["large_thumbnail"] = np.where(
        df["thumbnail"].isna(), "cover-not-found.jpg", df["thumbnail"] + "&fife=w800"
    )
    return df

def build_vector_store():
    docs = TextLoader("tagged_description.txt").load()
    chunks = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0).split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings())

books = load_data()
db = build_vector_store()
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All"] + list(TONE_MAP.keys())

# ── Core logic ────────────────────────────────────────────
def format_authors(raw: str) -> str:
    parts = raw.split(";")
    if len(parts) == 1: return raw
    return f"{', '.join(parts[:-1])}, and {parts[-1]}" if len(parts) > 2 else f"{parts[0]} and {parts[1]}"

def get_recommendations(query: str, category: str, tone: str) -> pd.DataFrame:
    ids = [int(r.page_content.strip('"').split()[0]) for r in db.similarity_search(query, k=TOP_K["initial"])]
    recs = books[books["isbn13"].isin(ids)]
    if category != "All":
        recs = recs[recs["simple_categories"] == category]
    recs = recs.head(TOP_K["final"])
    if tone in TONE_MAP:
        recs = recs.sort_values(by=TONE_MAP[tone], ascending=False)
    return recs

def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        gr.Warning("Please enter a description.")
        return []
    recs = get_recommendations(query, category, tone)
    return [
        (row["large_thumbnail"], f"{row['title']} by {format_authors(row['authors'])}: {' '.join(row['description'].split()[:30])}...")
        for _, row in recs.iterrows()
    ]

# ── UI helpers ────────────────────────────────────────────
def nav_html():
    links = "".join(f'<a{"class=active" if i==0 else ""}>{l}</a>' for i, l in enumerate(NAV_LINKS))
    return f"""
    <div id="navbar">
      <div class="logo"><span>Book</span>Lens</div>
      <div class="nav-links">{links}</div>
      <div class="nav-right">&#x1F50D; &#x1F514; <div class="avatar">A</div></div>
    </div>"""

def hero_html():
    h = HERO
    return f"""
    <div id="hero">
      <div class="hero-content">
        <div class="hero-pill">&#x1F525;&nbsp;#1 in Books Today</div>
        <div class="hero-title">{h["title"]}</div>
        <div class="hero-meta">
          <span>{h["author"]}</span><span class="dot"></span>
          <span>{h["genre"]}</span><span class="dot"></span>
          <span style="color:#e50914">&#9733; {h["rating"]}</span><span class="dot"></span>
          <span>{h["pages"]} pages</span>
        </div>
        <div class="hero-btns">
          <button class="btn-play">&#9654;&nbsp;Read now</button>
          <button class="btn-info">&#9432;&nbsp;More info</button>
        </div>
      </div>
      <div class="hero-rank">1</div>
    </div>"""

def catbar_html():
    pills = "".join(
        f'<button class="cat-pill{"active" if i==0 else ""}" onclick="setPill(this)">{c}</button>'
        for i, c in enumerate(["All"] + CATEGORIES_EXTRA)
    )
    return f"""
    <div id="cat-bar">{pills}</div>
    <script>
    function setPill(el){{document.querySelectorAll('.cat-pill').forEach(p=>p.classList.remove('active'));el.classList.add('active');}}
    </script>"""

def section_header(title: str) -> str:
    return f"""
    <div class="section-head">
      <div class="section-title">{title}</div>
      <div class="section-see-all">See all &rsaquo;</div>
    </div>"""

# ── Styles ────────────────────────────────────────────────
CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
.gradio-container{background:#0a0a0f!important;font-family:'Segoe UI',sans-serif!important;padding:0!important;max-width:100%!important}
footer{display:none!important}

#navbar{background:#0a0a0f;border-bottom:1px solid #ffffff0f;padding:0 40px;display:flex;align-items:center;height:52px;gap:28px}
.logo{font-size:18px;font-weight:800;color:#fff;letter-spacing:-0.5px;margin-right:12px}
.logo span{color:#e50914}
.nav-links{display:flex;gap:24px}
.nav-links a{font-size:13px;color:#888;cursor:pointer;text-decoration:none}
.nav-links a.active{color:#fff;font-weight:600}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:16px;color:#888;font-size:19px}
.avatar{width:30px;height:30px;border-radius:50%;background:#e50914;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff}

#hero{position:relative;height:280px;overflow:hidden;display:flex;align-items:flex-end;background:linear-gradient(135deg,#0e0820,#200c0c,#081a10,#071422,#1a1200)}
#hero::after{content:'';position:absolute;inset:0;background:linear-gradient(to right,#0a0a0f 35%,transparent 70%),linear-gradient(to top,#0a0a0f 12%,transparent 55%)}
.hero-content{position:relative;z-index:2;padding:0 40px 28px;max-width:55%}
.hero-pill{display:inline-flex;align-items:center;gap:5px;background:#e509141a;border:1px solid #e5091433;border-radius:20px;padding:3px 12px;font-size:10px;font-weight:700;color:#e50914;margin-bottom:12px;letter-spacing:.06em;text-transform:uppercase}
.hero-title{font-size:30px;font-weight:900;line-height:1.15;margin-bottom:6px;letter-spacing:-.5px;color:#fff}
.hero-meta{font-size:12px;color:#888;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.dot{width:3px;height:3px;border-radius:50%;background:#555;display:inline-block}
.hero-btns{display:flex;gap:10px}
.btn-play{background:#fff;color:#0a0a0f;border:none;border-radius:6px;padding:8px 20px;font-size:13px;font-weight:700;cursor:pointer}
.btn-info{background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.15);border-radius:6px;padding:8px 18px;font-size:13px;cursor:pointer}
.hero-rank{position:absolute;right:40px;bottom:16px;z-index:2;font-size:90px;font-weight:900;color:#fff;opacity:.06;line-height:1;letter-spacing:-6px}

#cat-bar{padding:10px 40px;display:flex;gap:8px;background:#0d0d12;border-bottom:1px solid #ffffff0a;overflow-x:auto;scrollbar-width:none}
.cat-pill{padding:5px 16px;border-radius:20px;font-size:12px;border:1px solid #ffffff15;color:#777;background:transparent;cursor:pointer;white-space:nowrap}
.cat-pill:hover{border-color:#ffffff30;color:#ccc}
.cat-pilactive,.cat-pill.active{background:#e50914;color:#fff;border-color:#e50914;font-weight:600}

#search-zone{padding:20px 40px 22px;background:#0d0d12;border-bottom:1px solid #ffffff0a}
#search-zone label span{font-size:10px!important;font-weight:700!important;color:#888!important;text-transform:uppercase!important;letter-spacing:.13em!important}
#search-zone textarea{background:#131320!important;border:1.5px solid #ff8c00!important;border-radius:8px!important;padding:12px 14px!important;color:#ddd!important;font-size:13px!important;resize:none!important}
#search-zone textarea:focus{outline:none!important;box-shadow:0 0 0 3px rgba(255,140,0,.12)!important}
#search-zone textarea::placeholder{color:#444!important}
#search-zone select{background:#131320!important;border:1.5px solid #ff8c00!important;border-radius:8px!important;padding:9px 12px!important;color:#bbb!important;font-size:12px!important;width:100%!important}
#search-btn button{background:#ff8c00!important;color:#0a0a0f!important;border:none!important;border-radius:8px!important;padding:10px 24px!important;font-size:13px!important;font-weight:800!important;height:40px!important;cursor:pointer!important}
#search-btn button:hover{background:#e07d00!important}

.section-head{display:flex;align-items:center;justify-content:space-between;padding:20px 40px 12px}
.section-title{font-size:14px;font-weight:700;color:#fff;display:flex;align-items:center;gap:10px}
.section-title::before{content:'';width:3px;height:15px;border-radius:2px;background:#e50914;display:inline-block}
.section-see-all{font-size:11px;color:#555;cursor:pointer}

#results-gallery .gr-gallery{display:grid!important;grid-template-columns:repeat(4,1fr)!important;gap:16px!important;padding:0 40px 24px!important}
.gr-gallery-item img{border-radius:8px!important;aspect-ratio:2/3!important;object-fit:cover!important;width:100%!important;transition:transform .2s!important}
.gr-gallery-item img:hover{transform:scale(1.03)!important}

#site-footer{text-align:center;color:#222230;font-size:11px;padding:12px 0 20px;border-top:1px solid #ffffff05}
"""

# ── App ───────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="BookLens") as dashboard:

    gr.HTML(nav_html() + hero_html() + catbar_html())

    with gr.Group(elem_id="search-zone"):
        query = gr.Textbox(label="📖  Describe a book you'd love", placeholder="A dark fantasy about redemption and found family...", lines=2, max_lines=2)
        with gr.Row():
            cat   = gr.Dropdown(choices=categories, label="⊞  Category",       value="All", scale=2)
            tone  = gr.Dropdown(choices=tones,      label="☺  Emotional tone",  value="All", scale=2)
            btn   = gr.Button("🔍  Find my books", elem_id="search-btn", scale=1)

    gr.HTML(section_header("Recommended for you"))
    output = gr.Gallery(label="", columns=4, rows=2, height="auto", show_label=False, elem_id="results-gallery", object_fit="cover", preview=True)

    gr.HTML('<div id="site-footer">Powered by OpenAI Embeddings &middot; ChromaDB &middot; LangChain</div>')

    for trigger in [btn.click, query.submit]:
        trigger(fn=recommend_books, inputs=[query, cat, tone], outputs=output)

if __name__ == "__main__":
    dashboard.launch()