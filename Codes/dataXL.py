import pandas as pd
import os

# simple function to load data from a pickle file
# may use it later to do some data filtering here rather than in main.py
def load_data_from_pkl(filepath, force_update=False):
    cleaned_filepath = filepath.replace('.pkl', '_cleaned.pkl')
    if os.path.exists(cleaned_filepath) and not force_update:
        # Load pre-cleaned data
        return pd.read_pickle(cleaned_filepath)
    
    df = pd.read_pickle(filepath)
    #also going to clean the data some here
    words_to_remove = [
     "me", "myself", "we", "our", "ours", "ourselves", "i" ,
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom",
    "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "hamed", "having",
    "do", "does", "did", "doing",
     "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", 
    "can", "will", "just", "don", "should", "now", "a"]
    pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'
    df['review_text'] = df['review_text'].str.replace(pattern, '', regex=True, case=False)
    df['review_text'] = df['review_text'].str.lower()
    df.to_pickle(cleaned_filepath)
    
    return df