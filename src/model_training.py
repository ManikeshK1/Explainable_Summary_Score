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
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model MSE: {mse:.4f}")
    print(f"Model R2 Score: {r2:.4f}")
    
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


def explain_prediction(model, explainer, feature_dict, max_score=5.0):
    """
    Explains a single prediction with:
      PART 1 — Plain English paragraph a student can understand
      PART 2 — Raw SHAP feature attributions for technical reference
    """
    # Convert dict to dataframe row
    feature_df = pd.DataFrame([feature_dict])

    # Predict score
    score = model.predict(feature_df)[0]

    # Get SHAP values
    shap_vals = explainer.shap_values(feature_df)
    shap_vals_row = shap_vals[0]   # shape (n_features,)

    # ══════════════════════════════════════════════════════════
    # PART 1 – Plain English Explanation
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  PREDICTED SCORE: {score:.2f} / {max_score:.1f}")
    print("=" * 60)
    print("\n📝 WHAT THIS SCORE MEANS  (Plain English)\n")
    english_explanation = _generate_plain_english_explanation(
        score, feature_dict, shap_vals_row, max_score=max_score
    )
    print(english_explanation)

    # ══════════════════════════════════════════════════════════
    # PART 2 – SHAP Feature Attributions (Technical Detail)
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("📊 SHAP FEATURE ATTRIBUTIONS  (Technical Detail)\n")
    print(f"  {'Feature':<28} {'Value':>10}   {'SHAP Impact':>12}   Direction")
    print(f"  {'-'*28}   {'-'*10}   {'-'*12}   {'-'*9}")
    for i, col in enumerate(feature_df.columns):
        val  = feature_df.iloc[0, i]
        shap = shap_vals_row[i]
        direction = "▲ raises score" if shap > 0 else "▼ lowers score"
        print(f"  {col:<28} {val:>10.4f}   {shap:>+12.4f}   {direction}")
    print("-" * 60)

    return score, shap_vals

if __name__ == "__main__":
    file_path = "C:/Users/deii/Desktop/cloud/mohler_dataset_edited.csv"
    try:
        # 1. Load data
        df = load_dataset(file_path)
        
        # For demonstration, subset the data to speed things up (full dataset takes time)
        print("Using subset of 200 samples for demonstration...")
        df_subset = df.head(200).copy()
        
        # 2. Extract anchors & generate Semantic Mapping Features
        df_featured = build_feature_dataset(df_subset)
        
        # 3. Train Model and Explainer
        model, explainer = train_and_save_model(df_featured)
        
        # 4. Demonstrate Explainability on the first row
        row = df_featured.iloc[0]
        sample_features = {
            'feat_avg_semantic': row['feat_avg_semantic'],
            'feat_max_semantic': row['feat_max_semantic'],
            'feat_anchors_covered': row['feat_anchors_covered'],
            'feat_avg_jaccard': row['feat_avg_jaccard'],
            'feat_avg_edit': row['feat_avg_edit']
        }
        
        explain_prediction(model, explainer, sample_features)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
