# Niche Music Recommendation System
### Debiasing Collaborative Filtering with Hybrid NCF & Smooth xQuAD

Music recommendation algorithms often suffer from **popularity bias**: they recommend top-chart hits over and over because popular songs have millions of data points, while great songs by smaller artists (the bottom 40% long-tail) rarely get shown.

In this project, I built a hybrid recommender using the **Spotify Million Playlist Dataset (MPD)** and track audio features to recommend relevant, high-quality niche songs without hurting recommendation accuracy.

---

## What the System Does

1. **Stage 1 (Hybrid NCF):** Trains a Neural Collaborative Filtering model combining two branches:
   * **GMF Branch:** Captures user-song interaction patterns.
   * **MLP Branch:** Learns relationships between users and 13 audio acoustic features (energy, acousticness, valence, danceability, tempo, etc.).
2. **Stage 2 (Smooth xQuAD Reranking):** Post-processes the Top-50 candidate songs from the NCF model. Instead of picking only the highest-scoring popular tracks, it penalizes over-played songs and boosts tracks that match the user's acoustic taste profile.

```
Playlists (User Tastes) + Track Audio Features
                     │
                     ▼
       Hybrid NCF Model (GMF + MLP)
                     │
                     ▼
          Top Candidate Songs
                     │
                     ▼
         Smooth xQuAD Reranker 
  (Balances relevance with novelty penalty)
                     │
                     ▼
       Top-10 Debiased Recommendations
```

---

## Benchmark Results

I evaluated the model on a **20% held-out test set (66,740 interactions across 4,759 users)**. 

Adding the Smooth xQuAD reranking step significantly increased both recommendation quality and catalog diversity compared to the baseline model:

| Metric | Baseline (NCF Only) | With Smooth xQuAD | Improvement |
| :--- | :---: | :---: | :---: |
| **Precision@10** | 0.0008 | **0.0012** | **+51.28%** |
| **Recall@10** | 0.0007 | **0.0010** | **+38.26%** |
| **NDCG@10** | 0.0009 | **0.0012** | **+33.88%** |
| **Catalog Diversity (Inverse Popularity)** | 0.1599 | **0.2062** | **+28.94%** |

*Note: In an initial Random Forest feature analysis, track popularity alone had an importance score of **60.8%**, which proved that standard models rely heavily on popularity unless explicitly debiased.*

---

## Dataset Details

* **Spotify Million Playlist Dataset (MPD):** Sampled 5,000 playlists (5 slices) containing **333,697 positive interactions** across **91,484 unique tracks**.
* **Audio Features Dataset:** 89,741 tracks with 13 continuous acoustic features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `mode`, `explicit`, `popularity`).
* **Train / Val / Test Split:**
  * Train: 68% (226,913 interactions)
  * Validation: 12% (40,044 interactions with early stopping)
  * Test: 20% (66,740 interactions)
* **Negative Sampling:** Used an 8:1 negative-to-positive ratio to give the binary cross-entropy loss negative training signals, resulting in ~2.02 million training pairs.

---

## Project Structure

```
├── final_mlpr.ipynb          # Main Jupyter Notebook (data prep, model training, eval)
├── detailed_feedback.csv     # Simulated 7-day user feedback cohort data
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, SciPy (Sparse CSR matrices)
* **ML & Evaluation:** Scikit-Learn (RandomForestClassifier, ranking metrics)
* **Visualization:** Matplotlib, Seaborn

---

## How to Run

1. Open `final_mlpr.ipynb` in Jupyter Notebook, VS Code, or Kaggle.
2. Install required packages:
   ```bash
   pip install numpy pandas scikit-learn tensorflow
   ```
3. Run all cells to process the dataset, train the Hybrid NCF network, and view evaluation metrics.
