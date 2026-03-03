import pandas as pd
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer, util
# Disable futurewarnings from huggingface/transformers
warnings.simplefilter(action='ignore', category=FutureWarning)

from anchor_extraction import load_dataset, process_dataset_anchors

class SemanticMapper:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def compute_jaccard_similarity(self, text1, text2):
        """Computes Jaccard similarity between two strings."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def compute_edit_distance_similarity(self, text1, text2):
        """Computes a normalized Levenshtein edit distance similarity."""
        import Levenshtein
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        distance = Levenshtein.distance(text1.lower(), text2.lower())
        max_len = max(len(text1), len(text2))
        
        # Returns 1.0 for identical strings, 0.0 for completely different
        return 1.0 - (distance / max_len)

    def evaluate_student_answer(self, student_answer, anchors):
        """
        Evaluates a student's answer against a list of desired anchors.
        Returns aggregated features.
        """
        if not isinstance(student_answer, str) or not student_answer.strip():
            # Return zeros for empty answers
            return {
                'avg_semantic_sim': 0.0,
                'max_semantic_sim': 0.0,
                'anchors_covered': 0.0,
                'avg_jaccard': 0.0,
                'avg_edit_sim': 0.0
            }
        
        if not anchors:
            # Handle case with no anchors
            return {
                'avg_semantic_sim': 0.0,
                'max_semantic_sim': 0.0,
                'anchors_covered': 0.0,
                'avg_jaccard': 0.0,
                'avg_edit_sim': 0.0
            }

        student_embedding = self.model.encode(student_answer, convert_to_tensor=True)
        anchor_embeddings = self.model.encode(anchors, convert_to_tensor=True)
        
        # Calculate cosine similarities between student answer and all anchors
        # The result is a 1 x num_anchors tensor
        cosine_scores = util.cos_sim(student_embedding, anchor_embeddings)[0].cpu().numpy()
        
        # Calculate lexical metrics against each anchor
        jaccard_scores = [self.compute_jaccard_similarity(student_answer, anchor) for anchor in anchors]
        edit_scores = [self.compute_edit_distance_similarity(student_answer, anchor) for anchor in anchors]
        
        # Define a threshold for considering an anchor "covered"
        # Since we compare a full sentence against a phrase, similarity might not be 0.99
        coverage_threshold = 0.35 
        
        features = {
            'avg_semantic_sim': float(np.mean(cosine_scores)),
            'max_semantic_sim': float(np.max(cosine_scores)),
            'anchors_covered': float(np.sum(cosine_scores >= coverage_threshold) / len(anchors)),
            'avg_jaccard': float(np.mean(jaccard_scores)),
            'avg_edit_sim': float(np.mean(edit_scores))
        }
        
        return features

def generate_features(df):
    """Generates features for the entire dataframe."""
    mapper = SemanticMapper()
    
    print("Generating semantic mapping features...")
    # Initialize lists to hold feature columns
    avg_sem = []
    max_sem = []
    coverage = []
    avg_jac = []
    avg_edit = []
    
    # Process each row
    total_rows = len(df)
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{total_rows}...")
            
        features = mapper.evaluate_student_answer(row['student_answer'], row['anchors'])
        
        avg_sem.append(features['avg_semantic_sim'])
        max_sem.append(features['max_semantic_sim'])
        coverage.append(features['anchors_covered'])
        avg_jac.append(features['avg_jaccard'])
        avg_edit.append(features['avg_edit_sim'])
        
    # Add new feature columns to the dataframe
    df['feat_avg_semantic'] = avg_sem
    df['feat_max_semantic'] = max_sem
    df['feat_anchors_covered'] = coverage
    df['feat_avg_jaccard'] = avg_jac
    df['feat_avg_edit'] = avg_edit
    
    print("Feature generation complete.")
    return df

if __name__ == "__main__":
    file_path = "C:/Users/deii/Desktop/cloud/mohler_dataset_edited.csv"
    try:
        # Load data
        df = load_dataset(file_path)
        
        # Take a sample for testing 
        df_sample = df.head(10).copy()
        
        # Step 1: Extract anchors from desired answers
        df_sample = process_dataset_anchors(df_sample)
        
        # Step 2: Generate Features mapping student_answer against anchors
        df_featured = generate_features(df_sample)
        
        # Display results
        features_to_show = ['score_avg', 'feat_avg_semantic', 'feat_max_semantic', 'feat_anchors_covered', 'feat_avg_jaccard']
        print("\nResults for sample data:")
        print(df_featured[['student_answer'] + features_to_show].head().to_string())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
