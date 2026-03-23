import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Load dataset
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    print(f"Dataset loaded with {len(df)} rows.")
    return df

# Initialize models
print("Initializing KeyBERT model...")
# Using a lightweight but effective model for keyphrase extraction
kw_extractor = KeyBERT('all-MiniLM-L6-v2')
print("KeyBERT model initialized.")

def extract_anchors(text, num_anchors=5):
    """
    Extracts keyphrases (anchors) from a given text using KeyBERT.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Extract keywords
    # stop_words='english' removes common words
    # keyphrase_ngram_range=(1, 3) extracts unigrams, bigrams, and trigrams
    keywords = kw_extractor.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3), 
        stop_words='english', 
        use_maxsum=True, 
        nr_candidates=20, 
        top_n=num_anchors
    )
    
    # Return just the extracted phrases (ignoring the confidence scores for now)
    return [kw[0] for kw in keywords]

def process_dataset_anchors(df, text_column='desired_answer'):
    """
    Processes the DataFrame to extract anchors for the specified column.
    """
    print(f"Extracting anchors for column: {text_column}...")
    df['anchors'] = df[text_column].apply(lambda x: extract_anchors(x) if pd.notnull(x) else [])
    print("Anchor extraction complete.")
    return df

if __name__ == "__main__":
    file_path = "C:/Users/deii/Desktop/cloud/mohler_dataset_edited.csv"
    try:
        df = load_dataset(file_path)
        
        # We only need one unique desired_answer per question to act as the reference
        # So group by question and desired_answer
        unique_references = df[['question', 'desired_answer']].drop_duplicates().reset_index(drop=True)
        print(f"Found {len(unique_references)} unique reference answers.")
        
        # Test on a few samples
        sample_refs = unique_references.head(5)
        processed_samples = process_dataset_anchors(sample_refs)
        
        for idx, row in processed_samples.iterrows():
            print(f"\nQuestion: {row['question']}")
            print(f"Reference Answer: {row['desired_answer']}")
            print(f"Extracted Anchors: {row['anchors']}")
            
    except Exception as e:
        print(f"Error: {e}")
