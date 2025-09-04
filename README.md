# MyRecEngine

**Personalized Product Recommendation Engine**

A compact recommendation engine built as part of the Flipkart Grid 5.0 submission. MyRecEngine combines a reinforcement-learning based category ranker (Thompson Sampling) with collaborative filtering, product-embedding similarity and co-visitation analytics to produce category- and product-level recommendations. A Streamlit UI (`app.py`) demonstrates the system interactively.

---

## Key features

* **Dynamic category ranking** using Thompson Sampling to balance exploration and exploitation.
* **Product recommendation** via item-item collaborative filtering (cosine similarity) over a user–product interaction matrix.
* **Similar-product retrieval** using precomputed product embeddings (cosine similarity).
* **Complementary-product suggestions** using a co-visitation matrix built from session sequences.
* **Streamlit demo** for interactive exploration of recommendations and internal data views.

---

## Tech stack

* Python 3.8+
* pandas, numpy
* scikit-learn (cosine similarity)
* scipy (sparse matrices)
* Streamlit (demo UI)

---

## Repository structure

```
MyRecEngine/
├── app.py                       # Streamlit demo & orchestration
├── final_data.csv               # Interaction dataset used by notebooks/app
├── articles.csv                 # Product metadata
├── productembed.txt             # Product embedding vectors (id -> vector)
├── markov.ipynb                 # Markov-model experiments (notebooks)
├── rl-category-ranking.ipynb    # Reinforcement-learning/category-ranking notebook
├── grid.pptx                    # Project pitch / slides
├── README.md                    # (this file)
└── LICENSE                      # MIT License
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lolzone13/MyRecEngine.git
cd MyRecEngine
```

2. Create a virtual environment and install dependencies (example using `venv`):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install the core packages manually:

```bash
pip install pandas numpy scikit-learn scipy streamlit
```

---

## Data

The project expects the following files (included in the repository):

* `final_data.csv` — main interaction dataset (user sessions, purchases, category sequences).
* `articles.csv` — product metadata (id, title, category, etc.).
* `productembed.txt` — product embeddings used for similarity lookups.

If you replace these with your own datasets, ensure the column names and ID formats match what `app.py` expects (article IDs, `category_sequence`, user/session identifiers, etc.).

---

## Running the demo

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

The app provides two main views: `recommendations` (interactive recommendation exploration) and `dataframe` (raw data inspection). The demo shows category recommendations, product lists for selected categories, similar products via embeddings, and co-visitation based complementary items.

---

## How the algorithms work (high level)

### 1) Category ranking — Thompson Sampling

* The category ranker treats each category as a bandit arm. It keeps track of successes and pulls and samples Beta distributions to estimate the probability that a category will produce a positive interaction.
* The system runs both a global (all users) simulation and a user-specific simulation. A weighted ensemble of these probabilities produces the final ranked categories, allowing personalization while preserving global popularity trends.

### 2) Product recommendations — Collaborative filtering (item-item)

* For each target category, the app builds a **user × product** interaction matrix restricted to items in that category.
* It computes cosine similarities between items (columns) using a sparse representation for memory efficiency.
* For the requesting user, the model computes a score for candidate items as the similarity-weighted sum of items the user has already interacted with/purchased, then returns the top‑N items the user hasn’t purchased.

### 3) Similar items & co-visitation

* Similar items are retrieved using precomputed embedding vectors: cosine similarity on embeddings surfaces nearest neighbors.
* A co-visitation matrix is built from session sequences by counting pairs of items appearing in the same session and applying positional weights; frequently co-visited/co-purchased pairs become complementary suggestions ("People who bought X also bought Y").

---

## Evaluation & Metrics

The repository and slides suggest these evaluation metrics and experiment ideas:

* **Click-through rate (CTR)** for recommended items
* **Conversion rate** of recommendations (how many become purchases)
* **Average Order Value (AOV)** lift from cross-sell suggestions
* **A/B testing** and user feedback (surveys) for real-world validation

---

## Notes & limitations

* The approach relies on sufficient historical interaction data; cold-start users/items will be harder to serve accurately.
* Collaborative filtering at scale can be computationally expensive—sparse matrices and blocking per-category are used to mitigate this.
* The Thompson Sampling implementation is lightweight and designed for prototyping; production-grade RL would need careful reward engineering, logging, and safety checks.

---

## Future work

* Introduce deep-learning models (image-based recommendations or Transformer-based ranking).
* Use Graph Neural Networks (GNNs) to model user–item interaction graphs and reduce cold-start problems.
* Region-aware trending and personalized deals.

