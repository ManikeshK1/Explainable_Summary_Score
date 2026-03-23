# ExplainGrade — Explainable Short Answer Grading System

> **AI-powered short answer grading that explains itself — in plain English.**

🌐 **[Live Demo → manikeshk1.github.io/Explainable_Summary_Score](https://manikeshk1.github.io/Explainable_Summary_Score/)**

---

## What Is This?

ExplainGrade is an Automated Short Answer Grading (ASAG) system that scores student text responses and — critically — **explains exactly why** the student received that score in plain language anyone can understand. No black-box predictions.

It was trained on the **Mohler Short Answer Grading Dataset** (2,200+ student responses to computer science exam questions graded 0–5 by human annotators).

---

## Two Ways to Use

### 🌐 Option 1 — Use Directly in the Browser (Zero Install)

**Everything runs in your browser. No data is sent to any server.**

| Feature | How to use |
|---|---|
| **Single Answer Grading** | Go to the [Live Demo](https://manikeshk1.github.io/Explainable_Summary_Score/#demo) → paste reference answer + student answer → click *Compute Score* |
| **Batch CSV Grading** | Go to [Batch Grade](https://manikeshk1.github.io/Explainable_Summary_Score/#batch) → upload a `.csv` with columns `question`, `desired_answer`, `student_answer` |
| **Class Evaluation** | Go to [Script Eval](https://manikeshk1.github.io/Explainable_Summary_Score/#script-eval) → upload a Google Meet transcript `.docx` + student summaries `.xlsx` |

> ⚠ **Limitation:** The browser uses TF-cosine similarity as a proxy for semantic similarity. For higher accuracy, use the local runner below.

---

### 🖥️ Option 2 — Run Locally (Better Performance & Accuracy)

Uses real **sentence-transformers** (`all-MiniLM-L6-v2`) for semantic similarity — more accurate than the browser approximation. Ideal for large classes or repeated use.

#### Prerequisites

- Python **3.9 or newer** → [Download Python](https://www.python.org/downloads/)
- Git → [Download Git](https://git-scm.com/downloads)

#### Step-by-step setup

```bash
# 1. Clone the repository
git clone https://github.com/ManikeshK1/Explainable_Summary_Score.git
cd Explainable_Summary_Score

# 2. Create a virtual environment
#    Windows:
python -m venv venv
venv\Scripts\activate

#    macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies
#    (downloads sentence-transformers, torch, python-docx, pandas, etc.)
pip install -r requirements.txt

# 4. Run the local grader
python local_grader.py \
  --docx "path/to/your/transcript.docx" \
  --xlsx "path/to/your/summaries.xlsx" \
  --max-score 5

# Optional: custom output file name
python local_grader.py \
  --docx "transcript.docx" \
  --xlsx "summaries.xlsx" \
  --max-score 10 \
  --output "class_results.csv"
```

> **First run only:** The script will download the `all-MiniLM-L6-v2` model (~90 MB). It is cached automatically after the first download.

#### Expected XLSX format

Your student summaries spreadsheet must have these columns (column names are matched flexibly):

| Column | Flexible names accepted |
|---|---|
| Student email | `emailAddress`, `email`, `mail` |
| Student name | `name`, `student_name` |
| Student summary | `summary`, `answer`, `response` |

#### Output

Results are saved as `grading_results.csv` (or the name you pass via `--output`) with columns:

```
email, name, final, stage1, stage2, semantic, jaccard, note
```

---

## Grading Pipeline

| Stage | What it does |
|---|---|
| **Stage 1 — Rule-Based Floor** | Word recall + bigram bonus — sets a guaranteed minimum score |
| **Stage 2 — NLP Similarity** | Jaccard · 15% + Edit · 5% + Cosine · 15% + Norm Word Count · 15% + Semantic · 50% |
| **Final Score** | `min(max_score, stage1 + stage2)` |

---

## Project Structure

```
.
├── docs/                  # GitHub Pages website (static)
│   ├── index.html         # Main UI page
│   ├── style.css          # Design system & styles
│   ├── scorer.js          # In-browser ASAG engine
│   └── app.js             # UI controller
├── src/                   # Python ML pipeline (model training)
│   ├── anchor_extraction.py
│   ├── semantic_mapping.py
│   └── model_training.py
├── local_grader.py        # ← Local CLI grader (new)
├── requirements.txt       # ← Python dependencies for local runner
├── mohler_dataset_edited.csv
└── README.md
```

---

## Python ML Pipeline (Model Training)

```bash
# Requires the same venv as above
python src/model_training.py
```

The script will:
1. Load `mohler_dataset_edited.csv`
2. Extract anchors from each desired answer using KeyBERT
3. Generate 5 features for every student answer
4. Train a Random Forest regressor and print evaluation metrics

---

## CSV Format for Batch Grading (Website)

```csv
question,desired_answer,student_answer
What is photosynthesis?,Plants use sunlight water and CO2 to make glucose.,Plants make food using sunlight and produce oxygen.
```

The website auto-detects flexible column names: `desired_answer` / `reference_answer` / `ideal_answer` and `student_answer` / `response`.

---

## References

- Ahmad Ayaan & Kok-Why Ng (2024). *Automated grading using natural language processing and semantic analysis.* PMC12171532.
- Filighera et al. (2023). *Our System for Short Answer Grading using Generative Models.* BEA Workshop, ACL 2023.
- Mohler et al. (2011). *Learning to grade short answer questions using semantic similarity measures and dependency graph alignments.* ACL.

---

## License

MIT — free to use, share, and build upon.
