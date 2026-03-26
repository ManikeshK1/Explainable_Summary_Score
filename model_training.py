import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

from anchor_extraction import load_dataset, process_dataset_anchors
from semantic_mapping import generate_features

def build_feature_dataset(df):
    """
    Extracts anchors and generates features for the dataset.
    Takes a while on the full 2200-row dataset due to transformer inference.
    """
    df = process_dataset_anchors(df)
    df = generate_features(df)
    return df

def train_and_save_model(df, model_save_path="asag_scoring_model.pkl", explainer_save_path="shap_explainer.pkl"):
    """
    Trains a predictive model on the generated features and saves it.
    """
    print("Preparing data for training...")
    # Features we generated
    feature_cols = ['feat_avg_semantic', 'feat_max_semantic', 
                    'feat_anchors_covered', 'feat_avg_jaccard', 'feat_avg_edit']
    
    # Target variable (normalized out of 5 usually in Mohler)
    target_col = 'score_avg'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training RandomForestRegressor on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model MSE: {mse:.4f}")
    print(f"Model R2 Score: {r2:.4f}")
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y_test, y_pred)
    print(f"Model Pearson r: {corr:.4f}")
    
    # Initialize and save SHAP explainer for future inference explainability
    print("Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Save the model and the explainer to disk
    print(f"Saving model to {model_save_path}...")
    joblib.dump(model, model_save_path)
    # joblib.dump(explainer, explainer_save_path) # SHAP explainers can be tricky to pickle, better to instantiate on the fly with the model
    
    print("Training pipeline complete.")
    return model, explainer

def _generate_plain_english_explanation(score, feature_dict, shap_vals_row, max_score=5.0):
    """
    Converts SHAP values and feature values into plain English sentences
    that a student can understand without any knowledge of machine learning.

    Parameters
    ----------
    score          : float  – model predicted score
    feature_dict   : dict   – raw feature values for this student
    shap_vals_row  : array  – per-feature SHAP values (same order as feature_dict)
    max_score      : float  – maximum possible score (default 5)

    Returns
    -------
    str : a multi-line plain English explanation
    """
    avg_sem   = feature_dict.get('feat_avg_semantic',    0.0)
    max_sem   = feature_dict.get('feat_max_semantic',    0.0)
    coverage  = feature_dict.get('feat_anchors_covered', 0.0)
    jaccard   = feature_dict.get('feat_avg_jaccard',     0.0)
    edit_sim  = feature_dict.get('feat_avg_edit',        0.0)

    feature_names = [
        'feat_avg_semantic', 'feat_max_semantic',
        'feat_anchors_covered', 'feat_avg_jaccard', 'feat_avg_edit'
    ]
    shap_map = dict(zip(feature_names, shap_vals_row))

    lines = []

    # ── Overall verdict ──────────────────────────────────────────────────────
    pct = score / max_score
    if pct >= 0.85:
        overall = (f"Your answer is excellent! You scored {score:.2f} out of {max_score:.1f}, "
                   f"placing you among the top performers.")
    elif pct >= 0.65:
        overall = (f"Your answer is good. You scored {score:.2f} out of {max_score:.1f}. "
                   f"You covered most of what was expected, with a little room to improve.")
    elif pct >= 0.40:
        overall = (f"Your answer is partially correct. You scored {score:.2f} out of {max_score:.1f}. "
                   f"You got some important points, but missed several key ideas.")
    else:
        overall = (f"Your answer needs significant improvement. You scored {score:.2f} out of {max_score:.1f}. "
                   f"The answer is missing most of the core concepts expected.")
    lines.append(overall)
    lines.append("")

    # ── Key concept coverage ─────────────────────────────────────────────────
    covered_pct = int(round(coverage * 100))
    shap_cov = shap_map.get('feat_anchors_covered', 0)
    if covered_pct >= 70:
        cov_sentence = (f"✔ You addressed approximately {covered_pct}% of the key concepts "
                        f"required in the ideal answer — this was a strong factor in your favour.")
    elif covered_pct >= 40:
        cov_sentence = (f"⚠ You addressed only about {covered_pct}% of the key concepts "
                        f"required. Missing the remaining concepts reduced your score "
                        f"(this factor {"raised" if shap_cov > 0 else "lowered"} your mark).")
    else:
        cov_sentence = (f"✘ You addressed very few ({covered_pct}%) of the key concepts expected "
                        f"in a complete answer. This was the biggest reason your score is low.")
    lines.append(cov_sentence)

    # ── Semantic / meaning similarity ────────────────────────────────────────
    shap_avg = shap_map.get('feat_avg_semantic', 0)
    shap_max = shap_map.get('feat_max_semantic', 0)
    if avg_sem >= 0.55:
        sem_sentence = (f"✔ The overall meaning of your answer closely matched the expected answer "
                        f"(semantic similarity: {avg_sem:.0%}). This shows you understood the topic well.")
    elif avg_sem >= 0.35:
        sem_sentence = (f"⚠ The meaning of your answer partially matched the expected answer "
                        f"(semantic similarity: {avg_sem:.0%}). You understood some concepts but your "
                        f"explanation could be more precise or complete.")
    else:
        sem_sentence = (f"✘ The meaning of your answer was quite different from the expected answer "
                        f"(semantic similarity: {avg_sem:.0%}). The grader could not identify the "
                        f"core idea in your response.")
    lines.append(sem_sentence)

    # Best-matching part of the answer
    if max_sem >= 0.65:
        lines.append(f"   Your best sentence or phrase was a strong match "
                     f"(peak similarity: {max_sem:.0%}).")
    elif max_sem >= 0.45:
        lines.append(f"   Your best phrase was a partial match "
                     f"(peak similarity: {max_sem:.0%}). Try to be more specific.")
    else:
        lines.append(f"   Even your closest phrase had a low match "
                     f"(peak similarity: {max_sem:.0%}) — try to use the correct terminology.")

    # ── Word-level overlap (Jaccard) ─────────────────────────────────────────
    shap_jac = shap_map.get('feat_avg_jaccard', 0)
    if jaccard >= 0.30:
        jac_sentence = (f"✔ You used many of the same key words as the model answer "
                        f"(word overlap: {jaccard:.0%}), which helped your score.")
    elif jaccard >= 0.15:
        jac_sentence = (f"⚠ You used some of the expected vocabulary (word overlap: {jaccard:.0%}), "
                        f"but using more subject-specific terms would improve your mark.")
    else:
        jac_sentence = (f"✘ Very few of the key words from the model answer appeared in your response "
                        f"(word overlap: {jaccard:.0%}). Make sure to use the correct terminology.")
    lines.append(jac_sentence)

    # ── Phrasing / edit distance ─────────────────────────────────────────────
    shap_edit = shap_map.get('feat_avg_edit', 0)
    if edit_sim >= 0.55:
        edit_sentence = (f"✔ The way you phrased your answer was very similar to the expected "
                         f"answer (phrasing similarity: {edit_sim:.0%}).")
    elif edit_sim >= 0.30:
        edit_sentence = (f"➜ Your phrasing was somewhat similar to the model answer "
                         f"(phrasing similarity: {edit_sim:.0%}). Consider restructuring your "
                         f"sentences to be more concise and on-point.")
    else:
        edit_sentence = (f"✘ Your phrasing was quite different from the expected answer "
                         f"(phrasing similarity: {edit_sim:.0%}). This may indicate that you "
                         f"expressed the idea in an unrelated way or went off-topic.")
    lines.append(edit_sentence)

    # ── How to improve ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("💡 How to improve:")
    if covered_pct < 70:
        lines.append("   • Re-read the question carefully and make sure you address ALL required points.")
    if avg_sem < 0.50:
        lines.append("   • Focus on expressing the core idea more clearly and directly.")
    if jaccard < 0.25:
        lines.append("   • Use domain-specific vocabulary and keywords from your notes/textbook.")
    if edit_sim < 0.40:
        lines.append("   • Try to write more structured, concise sentences that match the question's scope.")

    return "\n".join(lines)


STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall','can',
    'to','of','in','for','on','with','at','by','from','as','into','through',
    'and','but','or','nor','so','yet','both','either','neither','not','no',
    'that','this','these','those','which','who','what','how','when','where','why',
    'if','because','while','although','though','since','unless','until','than',
    'i','you','he','she','it','we','they','me','him','her','us','them',
    'my','your','his','its','our','their','all','each','any','more','most',
    'other','some','only','very','just','s','t','re','ve'
}

def _content_tokens(text):
    """Lowercase words, strip punctuation, remove stopwords."""
    import re
    words = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
    return [w for w in words if w and w not in STOPWORDS]

def _all_tokens(text):
    """Same but keeping stopwords — used for student side (broad match)."""
    import re
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()

# ─────────────────────────────────────────────────────────────
# STAGE 1 — Rule-Based Direct Word Match (floor / minimum)
# ─────────────────────────────────────────────────────────────

def rule_based_score(reference_answer: str, student_answer: str, max_score: float = 5.0) -> float:
    if not reference_answer.strip() or not student_answer.strip():
        return 0.0

    # Normalization
    import re
    def normalise(t):
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', t.lower())).strip()

    # Stage 1A: Content Word Recall (Unigrams)
    ref_words = list(dict.fromkeys(_content_tokens(reference_answer)))
    if not ref_words: return 0.0
    stu_word_set = set(_all_tokens(student_answer))
    word_recall = sum(1 for w in ref_words if w in stu_word_set) / len(ref_words)

    # Stage 1B: Bigram Recall Bonus (+15% weights)
    # Matching "neural network" is worth more than just "neural" and "network" separately
    ref_content = _content_tokens(reference_answer)
    ref_bigrams = [f"{ref_content[i]} {ref_content[i+1]}" for i in range(len(ref_content)-1)]
    stu_norm = normalise(student_answer)
    bigram_hits = sum(1 for bg in ref_bigrams if bg in stu_norm) if ref_bigrams else 0
    bigram_bonus = (bigram_hits / len(ref_bigrams) * 0.15) if ref_bigrams else 0

    return min(1.0, word_recall + bigram_bonus) * max_score

def tfidf_cosine_sim(text1: str, text2: str) -> float:
    """Computes TF-IDF Cosine Similarity between two texts.
    NOTE: Only used for anchor-extraction features.
    For the paper's grading (Sc, Stf) use _tf_cosine_sim — see below.
    """
    import math
    toks1 = _content_tokens(text1)
    toks2 = _content_tokens(text2)
    if not toks1 or not toks2:
        return 0.0
    
    def build_tf(toks):
        import collections
        freq = collections.Counter(toks)
        tot = len(toks) or 1
        return {k: v / tot for k, v in freq.items()}
    
    tf1, tf2 = build_tf(toks1), build_tf(toks2)
    all_terms = set(tf1.keys()).union(set(tf2.keys()))
    
    idf = {}
    for t in all_terms:
        doc_count = (1 if t in tf1 else 0) + (1 if t in tf2 else 0)
        idf[t] = math.log(3 / (1 + doc_count))
        
    dot, mag1, mag2 = 0.0, 0.0, 0.0
    for t in all_terms:
        v1 = tf1.get(t, 0) * idf[t]
        v2 = tf2.get(t, 0) * idf[t]
        dot += v1 * v2
        mag1 += v1 * v1
        mag2 += v2 * v2
        
    if mag1 == 0 or mag2 == 0:
        return 0.0
    val = dot / (math.sqrt(mag1) * math.sqrt(mag2))
    return max(0.0, min(1.0, val))


def _tf_cosine_sim(text1: str, text2: str) -> float:
    """
    Plain TF cosine similarity (NO IDF weighting).

    Used for the paper's Sc (cosine similarity) and Stf (semantic proxy).

    Why not TF-IDF here:
      With only 2 documents in the corpus, IDF = log(3/(1+count)).
      For any word appearing in BOTH documents: IDF = log(3/3) = 0.
      This zeros out all shared words — so two very similar answers
      score ~0 and trigger the paper's F=0 rule (Stf<0.2).
      Plain TF cosine correctly rewards shared vocabulary.
    """
    import math
    import collections

    toks1 = _all_tokens(text1)
    toks2 = _all_tokens(text2)
    if not toks1 or not toks2:
        return 0.0

    tf1 = collections.Counter(toks1)
    tf2 = collections.Counter(toks2)
    all_terms = set(tf1) | set(tf2)

    dot  = sum(tf1.get(t, 0) * tf2.get(t, 0) for t in all_terms)
    mag1 = math.sqrt(sum(v * v for v in tf1.values()))
    mag2 = math.sqrt(sum(v * v for v in tf2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag1 * mag2)))


# ─────────────────────────────────────────────────────────────
# PAPER GRADING HELPERS
# Paper: "Automated grading using NLP and semantic analysis"
# PMC12171532  —  Ahmad Ayaan & Kok-Why Ng
# ─────────────────────────────────────────────────────────────

def _jaccard_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity between two texts using all (non-stopword) tokens."""
    set1 = set(_all_tokens(text1))
    set2 = set(_all_tokens(text2))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _edit_similarity(text1: str, text2: str) -> float:
    """
    Normalized character-level edit similarity.
    Se = 1 - (edit_distance / max_char_length)
    The paper uses the inverted form (Se)^-1 which equals this value.
    """
    try:
        import Levenshtein
        a, b = text1.lower(), text2.lower()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        dist = Levenshtein.distance(a, b)
        return 1.0 - dist / max(len(a), len(b))
    except ImportError:
        # Pure-Python fallback (no Levenshtein library)
        a, b = text1.lower(), text2.lower()
        longer = max(len(a), len(b))
        if longer == 0:
            return 1.0
        common = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        return common / longer


def _normalized_word_count(reference_answer: str, student_answer: str) -> float:
    """
    Sw: reference keyword count / student keyword count, capped at 1.
    Paper: "calculated by dividing the sample answer's keywords
            by the student's answer's keywords."
    """
    ref_kw = _content_tokens(reference_answer)
    stu_kw = _content_tokens(student_answer)
    if not stu_kw:
        return 0.0
    return min(1.0, len(ref_kw) / len(stu_kw))


# ─────────────────────────────────────────────────────────────
# STAGE 2 — Paper's Grading Method (PMC12171532)
# ─────────────────────────────────────────────────────────────

def paper_grading_score(reference_answer: str, student_answer: str,
                        max_score: float = 5.0) -> float:
    """
    Implements the EXACT grading method from the paper (PMC12171532).

    Weights:
      Jaccard (wj)          = 0.15
      Edit Similarity (we)  = 0.05   [(Se)^-1 inverted form]
      Cosine (wc)           = 0.15
      Norm. Word Count (ww) = 0.15
      Semantic / USE (wtf)  = 0.50   [approximated with TF-IDF cosine]

    Equations:
      Cnlp = min(max(0, wj·Sj + we·Se + wc·Sc + ww·Sw), 1)      ...(a)
      C    = min(max(0, wtf·Stf + (1-wtf)·Cnlp), 1)              ...(b)
      F    = { 0  if Stf < 0.2                                    ...(c)
             { 1  if Stf >= 0.9 AND Sw >= 0.85
             { C  otherwise
      M    = ceil(min(F · T, T))                                  ...(d)

    Returns F * max_score (floating-point) so it can be summed
    with the rule-based stage using the existing additive logic.
    """
    import math
    if not reference_answer.strip() or not student_answer.strip():
        return 0.0

    Sj  = _jaccard_similarity(reference_answer, student_answer)      # wj = 0.15
    Se  = _edit_similarity(reference_answer, student_answer)          # we = 0.05
    Sc  = _tf_cosine_sim(reference_answer, student_answer)            # wc = 0.15  (plain TF cosine)
    Sw  = _normalized_word_count(reference_answer, student_answer)    # ww = 0.15
    Stf = _tf_cosine_sim(reference_answer, student_answer)            # wtf= 0.50  (USE proxy via TF cosine)

    # Equation (a): Combined NLP base score
    Cnlp = min(1.0, max(0.0, 0.15*Sj + 0.05*Se + 0.15*Sc + 0.15*Sw))

    # Equation (b): Confidence score
    C = min(1.0, max(0.0, 0.50*Stf + 0.50*Cnlp))

    # Equation (c): Final score with threshold rules
    if Stf < 0.2:
        F = 0.0
    elif Stf >= 0.9 and Sw >= 0.85:
        F = 1.0
    else:
        F = C

    return F * max_score


# ─────────────────────────────────────────────────────────────
# TWO-STAGE PREDICTION
# ─────────────────────────────────────────────────────────────

# def three_stage_predict(model, reference_answer: str, student_answer: str,
#                         feature_dict: dict, max_score: float = 5.0) -> dict:
#     """
#     Returns:
#         stage1_score : rule-based word-match floor
#         stage2_score : paper's NLP + semantic grading (PMC12171532)
#         final_score  : min(max_score, stage1 + stage2)
#     """
#     stage1 = rule_based_score(reference_answer, student_answer, max_score)
#     stage2 = paper_grading_score(reference_answer, student_answer, max_score)
#     final  = min(max_score, stage1 + stage2)
#     return {
#         'stage1_score': stage1,
#         'stage2_score': stage2,
#         'final_score':  final
#     }

# C:\Users\deii\Desktop\cloud\model_training.py

def three_stage_predict(model, reference_answer, student_answer, feature_dict, max_score=5.0):
    """
    Improved Two-Stage Pipeline with Human-Alignment Calibration.
    """
    # 1. Calculate raw components
    stage1 = rule_based_score(reference_answer, student_answer, max_score)
    stage2_raw = paper_grading_score(reference_answer, student_answer, max_score)
    
    # 2. CALIBRATION (The "Score Booster")
    # We apply a slight non-linear boost to Stage 2. 
    # This pushes scores in the 2-3 range slightly higher to match human averages.
    ratio = stage2_raw / max_score
    calibrated_stage2 = (ratio ** 0.75) * max_score # Power transform pushes low scores up
    
    # 3. Aggregation
    # Stage 1 acts as the floor, Stage 2 provides the "knowledge" points
    combined = stage1 + (calibrated_stage2 * 0.6) # Weighted blend for better Pearson r
    
    final = min(max_score, combined)
    
    return {
        'stage1_score': stage1,
        'stage2_score': calibrated_stage2,
        'final_score':  final
    }
    
def _generate_plain_english_explanation(score, feature_dict, shap_vals_row, max_score=5.0):
    """
    Converts SHAP values and feature values into plain English sentences
    that a student can understand without any knowledge of machine learning.
    """
    avg_sem   = feature_dict.get('feat_avg_semantic',    0.0)
    max_sem   = feature_dict.get('feat_max_semantic',    0.0)
    coverage  = feature_dict.get('feat_anchors_covered', 0.0)
    jaccard   = feature_dict.get('feat_avg_jaccard',     0.0)
    edit_sim  = feature_dict.get('feat_avg_edit',        0.0)

    feature_names = [
        'feat_avg_semantic', 'feat_max_semantic',
        'feat_anchors_covered', 'feat_avg_jaccard', 'feat_avg_edit'
    ]
    shap_map = dict(zip(feature_names, shap_vals_row))
    lines = []

    pct = score / max_score
    if pct >= 0.85:
        overall = (f"Your answer is excellent! You scored {score:.2f} out of {max_score:.1f}, "
                   f"placing you among the top performers.")
    elif pct >= 0.65:
        overall = (f"Your answer is good. You scored {score:.2f} out of {max_score:.1f}. "
                   f"You covered most of what was expected, with a little room to improve.")
    elif pct >= 0.40:
        overall = (f"Your answer is partially correct. You scored {score:.2f} out of {max_score:.1f}. "
                   f"You got some important points, but missed several key ideas.")
    else:
        overall = (f"Your answer needs significant improvement. You scored {score:.2f} out of {max_score:.1f}. "
                   f"The answer is missing most of the core concepts expected.")
    lines.append(overall)
    lines.append("")

    covered_pct = int(round(coverage * 100))
    shap_cov = shap_map.get('feat_anchors_covered', 0)
    if covered_pct >= 70:
        cov_sentence = (f"✔ You addressed approximately {covered_pct}% of the key concepts "
                        f"required in the ideal answer — this was a strong factor in your favour.")
    elif covered_pct >= 40:
        direction = "raised" if shap_cov > 0 else "lowered"
        cov_sentence = (f"⚠ You addressed only about {covered_pct}% of the key concepts "
                        f"required. Missing the remaining concepts {direction} your mark.")
    else:
        cov_sentence = (f"✘ You addressed very few ({covered_pct}%) of the key concepts expected "
                        f"in a complete answer. This was the biggest reason your score is low.")
    lines.append(cov_sentence)

    if avg_sem >= 0.55:
        sem_sentence = (f"✔ The overall meaning of your answer closely matched the expected answer "
                        f"(semantic similarity: {avg_sem:.0%}). This shows you understood the topic well.")
    elif avg_sem >= 0.35:
        sem_sentence = (f"⚠ The meaning of your answer partially matched the expected answer "
                        f"(semantic similarity: {avg_sem:.0%}). Your explanation could be more precise.")
    else:
        sem_sentence = (f"✘ The meaning of your answer was quite different from the expected answer "
                        f"(semantic similarity: {avg_sem:.0%}). The grader could not identify the core idea.")
    lines.append(sem_sentence)

    if max_sem >= 0.65:
        lines.append(f"   Your best phrase was a strong match (peak similarity: {max_sem:.0%}).")
    elif max_sem >= 0.45:
        lines.append(f"   Your best phrase was a partial match (peak similarity: {max_sem:.0%}). Try to be more specific.")
    else:
        lines.append(f"   Even your closest phrase had low match (peak similarity: {max_sem:.0%}) — try correct terminology.")

    if jaccard >= 0.30:
        jac_sentence = (f"✔ You used many of the same key words as the model answer (word overlap: {jaccard:.0%}).")
    elif jaccard >= 0.15:
        jac_sentence = (f"⚠ You used some expected vocabulary (word overlap: {jaccard:.0%}), but more subject-specific terms would help.")
    else:
        jac_sentence = (f"✘ Very few key words from the model answer appeared in your response (word overlap: {jaccard:.0%}).")
    lines.append(jac_sentence)

    if edit_sim >= 0.55:
        edit_sentence = f"✔ The way you phrased your answer was very similar to the expected answer (phrasing similarity: {edit_sim:.0%})."
    elif edit_sim >= 0.30:
        edit_sentence = (f"➜ Your phrasing was somewhat similar (phrasing similarity: {edit_sim:.0%}). "
                         f"Consider restructuring your sentences to be more concise.")
    else:
        edit_sentence = (f"✘ Your phrasing was quite different from the expected answer (phrasing similarity: {edit_sim:.0%}). "
                         f"This may indicate you expressed ideas in an unrelated way or went off-topic.")
    lines.append(edit_sentence)

    lines.append("")
    lines.append("💡 How to improve:")
    if covered_pct < 70: lines.append("   • Re-read the question carefully and address ALL required points.")
    if avg_sem < 0.50:   lines.append("   • Focus on expressing the core idea more clearly and directly.")
    if jaccard < 0.25:   lines.append("   • Use domain-specific vocabulary and keywords from your notes/textbook.")
    if edit_sim < 0.40:  lines.append("   • Write more structured, concise sentences that match the question's scope.")

    return "\n".join(lines)


def explain_prediction(model, explainer, feature_dict,
                       reference_answer: str = "", student_answer: str = "",
                       max_score: float = 5.0):
    """
    Full three-stage prediction explanation:
      STAGE 1 — Rule-based word-match
      STAGE 2 — Semantic / AI (Random Forest)
      STAGE 3 — Complete sentence context
      FINAL   — Capped Sum

    Outputs:
      PART 1 — Plain English explanation for the student
      PART 2 — SHAP feature attributions for technical reference
    """
    feature_df    = pd.DataFrame([feature_dict])
    shap_vals     = explainer.shap_values(feature_df)
    shap_vals_row = shap_vals[0]

    # ── Two-stage scoring (paper method) ────────────────────────
    scores = three_stage_predict(model, reference_answer, student_answer,
                                 feature_dict, max_score)
    stage1 = scores['stage1_score']
    stage2 = scores['stage2_score']
    final  = scores['final_score']

    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(f"  SCORE BREAKDOWN  (PMC12171532 Grading Method)")
    print("=" * 65)
    print(f"  ⚖️  Stage 1 · Rule-Based (word match)         : +{stage1:.2f}")
    print(f"  📄 Stage 2 · Paper NLP + Semantic Grade       : +{stage2:.2f}")
    print(f"            (Cnlp=Jaccard·0.15+Edit·0.05")
    print(f"                  +Cosine·0.15+NormWC·0.15)")
    print(f"            (C   =0.5·Stf + 0.5·Cnlp)")
    print(f"  {'─' * 45}")
    print(f"  🏆 Final Score (capped at {max_score:.0f})              :  {final:.2f} / {max_score}")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════
    # PART 1 – Plain English
    # ══════════════════════════════════════════════════════════
    print("\n📝 WHAT THIS SCORE MEANS  (Plain English)\n")
    english_explanation = _generate_plain_english_explanation(
        final, feature_dict, shap_vals_row, max_score=max_score
    )
    print(english_explanation)

    # ══════════════════════════════════════════════════════════
    # PART 2 – SHAP Feature Attributions
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("📊 SHAP FEATURE ATTRIBUTIONS  (Technical Detail)\n")
    print(f"  {'Feature':<28} {'Value':>10}   {'SHAP Impact':>12}   Direction")
    print(f"  {'-'*28}   {'-'*10}   {'-'*12}   {'-'*9}")
    for i, col in enumerate(feature_df.columns):
        val  = feature_df.iloc[0, i]
        shap = shap_vals_row[i]
        direction = "▲ raises score" if shap > 0 else "▼ lowers score"
        print(f"  {col:<28} {val:>10.4f}   {shap:>+12.4f}   {direction}")
    print("-" * 65)

    return final, shap_vals

if __name__ == "__main__":
    file_path = "C:/Users/deii/Desktop/cloud/mohler_dataset_edited.csv"
    try:
        # 1. Load data
        df = load_dataset(file_path)

        # For demonstration, subset the data to speed things up
        print("Using subset of all samples for demonstration...")
        df_subset = df.copy()

        # 2. Extract anchors & generate Semantic Mapping Features
        df_featured = build_feature_dataset(df_subset)

        # 3. Train Model and Explainer
        model, explainer = train_and_save_model(df_featured)

        # 4. Demonstrate three-stage explainability on the first row
        row = df_featured.iloc[0]
        sample_features = {
            'feat_avg_semantic':    row['feat_avg_semantic'],
            'feat_max_semantic':    row['feat_max_semantic'],
            'feat_anchors_covered': row['feat_anchors_covered'],
            'feat_avg_jaccard':     row['feat_avg_jaccard'],
            'feat_avg_edit':        row['feat_avg_edit']
        }

        # Pass the original text so Stage 1 (rule-based) can be computed
        explain_prediction(
            model, explainer, sample_features,
            reference_answer=str(row.get('desired_answer', '')),
            student_answer=str(row.get('student_answer', '')),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
