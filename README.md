# Explainable Summary Score (ESS)

> **AI-powered short answer grading that explains itself — in plain English.**

🌐 **[Live Demo → manikeshk1.github.io/Explainable_Summary_Score](https://manikeshk1.github.io/Explainable_Summary_Score/)**

---

## What Is This?

ESS is an Automated Short Answer Grading (ASAG) system that scores student text responses and — critically — **explains exactly why** the student received that score, using plain language that anyone can understand. No black-box predictions.

It was trained on the **Mohler Short Answer Grading Dataset** (2,200+ student responses to computer science exam questions graded 0–5 by human annotators).

---

## Features

| Feature | Description |
|---|---|
| 🔑 **Anchor Extraction** | KeyBERT extracts the top 5 key concepts from the ideal answer |
| 🧬 **Semantic Mapping** | Cosine similarity between student answer and each anchor via `all-MiniLM-L6-v2` |
| 📊 **SHAP Explainability** | Per-feature attribution values — know exactly what raised or lowered the score |
| 📝 **Plain-English Explanation** | Student-friendly verdict + actionable "how to improve" tips |
| 📂 **Batch CSV Grading** | Upload a full class CSV and get scores for every row instantly |
| 🌗 **Dark / Light Mode** | Beautiful, professional UI with full theme switching |
| 🔒 **100% Private** | All scoring runs in your browser — no data is ever sent to a server |

---

## The 5 Features

| Feature Name | What It Measures | Model Weight |
|---|---|---|
| `feat_anchors_covered` | % of key concepts from ideal answer addressed | **35%** |
| `feat_avg_semantic` | Average cosine similarity of student answer to anchors | **30%** |
| `feat_max_semantic` | Peak semantic similarity (student's best phrase) | **18%** |
| `feat_avg_jaccard` | Word-set overlap — rewards correct terminology | **10%** |
| `feat_avg_edit` | Normalised Levenshtein phrasing similarity | **7%** |

---

## Local Setup (Python Pipeline)

```bash
# 1. Clone the repo
git clone https://github.com/ManikeshK1/Explainable_Summary_Score.git
cd Explainable_Summary_Score

# 2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux

# 3. Install dependencies
pip install pandas numpy scikit-learn sentence-transformers keybert shap python-levenshtein joblib

# 4. Run the full pipeline
python src/model_training.py
```

The script will:
1. Load `mohler_dataset_edited.csv`
2. Extract anchors from each desired answer
3. Generate the 5 features for every student answer
4. Train a Random Forest regressor and print evaluation metrics
5. Demonstrate an explainable prediction on the first row

---

## CSV Format for Batch Grading (Website)

```csv
question,desired_answer,student_answer
What is photosynthesis?,Plants use sunlight water and CO2 to make glucose.,Plants make food using sunlight and produce oxygen.
```

The website auto-detects flexible column names: `desired_answer` / `reference_answer` / `ideal_answer` and `student_answer` / `response`.

---

## Project Structure

```
.
├── docs/                  # GitHub Pages website (static)
│   ├── index.html
│   ├── style.css
│   ├── scorer.js          # In-browser ASAG engine
│   └── app.js             # UI controller
├── src/                   # Python pipeline source
│   ├── anchor_extraction.py
│   ├── semantic_mapping.py
│   └── model_training.py
├── mohler_dataset_edited.csv
└── README.md
```

---

## References

- Mohler et al. (2011). *Learning to grade short answer questions using semantic similarity measures and dependency graph alignments.* ACL.
- Saha et al. (2023). *System for short-answer grading using generative models.* BEA Workshop, ACL 2023.
- Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS.

---

## License

MIT — free to use, share, and build upon.
