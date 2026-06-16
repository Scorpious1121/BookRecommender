import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ── Load data ──────────────────────────────────────────────
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(), "cover-not-found.jpg", books["thumbnail"] + "&fife=w800"
)

# ── Load vector store ──────────────────────────────────────
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("books")
model = SentenceTransformer("all-MiniLM-L6-v2")

TONE_MAP = {
    "happy": "joy",
    "surprising": "surprise",
    "angry": "anger",
    "suspenseful": "fear",
    "sad": "sadness"
}

# ── Recommend function ─────────────────────────────────────
def recommend(query, category="All", tone="All", top_k=10):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=50)
    ids = [int(r.split()[0].strip('"')) for r in results["documents"][0]]

    recs = books[books["isbn13"].isin(ids)].copy()

    if category != "All":
        recs = recs[recs["simple_categories"] == category]

    if tone.lower() in TONE_MAP:
        recs = recs.sort_values(by=TONE_MAP[tone.lower()], ascending=False)

    return recs.head(top_k)

# ── Terminal display ───────────────────────────────────────
def display(recs, query, category, tone):
    W = 72
    print("\n" + "═" * W)
    print(f"  SEMANTIC BOOK RECOMMENDER".center(W))
    print("═" * W)
    print(f"  Query    : {query[:60]}")
    print(f"  Category : {category}   |   Tone : {tone}")
    print(f"  Results  : {len(recs)} books found")
    print("─" * W)

    for i, (_, row) in enumerate(recs.iterrows(), 1):
        # Authors
        parts = str(row["authors"]).split(";")
        authors = f"{parts[0]} and {parts[1]}" if len(parts) == 2 else \
                  f"{', '.join(parts[:-1])}, and {parts[-1]}" if len(parts) > 2 else parts[0]

        # Emotion bar
        emotions = {"joy": row.get("joy", 0), "fear": row.get("fear", 0),
                    "sadness": row.get("sadness", 0), "anger": row.get("anger", 0),
                    "surprise": row.get("surprise", 0)}
        top_emotion = max(emotions, key=emotions.get)
        score = emotions[top_emotion]
        bar = "█" * int(score * 12) + "░" * (12 - int(score * 12))

        # Description snippet
        desc = str(row["description"])
        snippet = desc[:250] + "..." if len(desc) > 120 else desc

        print(f"\n  [{i:02d}]  {row['title']}")
        print(f"        {authors}")
        print(f"        Category : {row.get('simple_categories', 'N/A')}   |   Rating : {row.get('average_rating', 'N/A')}")
        print(f"        Emotion  : {top_emotion.upper()} {bar} {score:.2f}")
        print(f"        {snippet}")
        print("  " + "·" * (W - 2))

    print("\n" + "═" * W + "\n")

# ── Main loop ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 72)
    print("  BookLens — Semantic Book Recommender  (type 'quit' to exit)".center(72))
    print("=" * 72)

    categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
    tones = ["All"] + list(TONE_MAP.keys())

    while True:
        print(f"\n  Categories : {', '.join(categories)}")
        print(f"  Tones      : {', '.join(tones)}\n")

        query = input("  Enter description: ").strip()
        if query.lower() == "quit":
            print("\n  Goodbye!\n")
            break

        category = input(f"  Category [{'/'.join(categories)}] (Enter = All): ").strip() or "All"
        tone = input(f"  Tone [{'/'.join(tones)}] (Enter = All): ").strip() or "All"

        recs = recommend(query, category, tone)
        display(recs, query, category, tone)

if __name__ == "__main__":
    main()