import pandas as pd
import os
import sys
from dataXL import load_data_from_pkl

#loading words in from opinion lexicon, used AI to do this part
def load_word_list(file_path):
    """Load words from a text file into a set, skipping comment lines"""
    words = set()
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comment lines starting with ;
                if line and not line.startswith(';'):
                    words.add(line.lower())
    except UnicodeDecodeError:
        # Fall back to latin-1 encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comment lines starting with ;
                if line and not line.startswith(';'):
                    words.add(line.lower())
    return words


def ratingSearch(query, df, positive_words, negative_words):
    #splitting it here instead of sending it from main, idk was less conusing for this one lol
    parts = query.split(":")
    aspect = parts[0].strip().lower()
    opinion = parts[1].strip().lower()
    
    # Split into individual words
    aspect_words = aspect.split()
    opinion_words = opinion.split()
    
    #looking at positive and negative words, making a list of opinion words from these two to rate how positive
    is_positive = any(word in positive_words for word in opinion_words)
    is_negative = any(word in negative_words for word in opinion_words)

    print(" Rating Search")
    #takes the review rating from other people and filters them to only be high ones if the opinion is positive, and if opinion leans towards negative then it filters for lower ratings
    if is_positive:
        print(f"Opinion is POSITIVE ")
        filtered_df = df[pd.to_numeric(df['customer_review_rating']) > 3]
    elif is_negative:
        print(f"Opinion is NEGATIVE ")
        filtered_df = df[pd.to_numeric(df['customer_review_rating']) <= 3]
    #just updates filtered to only include verified purchases, otherwise they might have never had the product
    filtered_df = filtered_df[filtered_df['amazon_verified_purchase'] == '1']
    #checking that the helpful count is above average as well, might just change this to median later
    avg_helpful_count = filtered_df['helpful_count'].astype(float).mean()

    filtered_df = filtered_df[filtered_df['helpful_count'].astype(float) >= avg_helpful_count]
    #basically these just do similar to the boolean, making the two dataframes for and and or matches, making it a string lower and that it contains the aspect/opinion
    aspect_match = filtered_df['review_text'].astype(str).str.lower().str.contains('|'.join(aspect_words), case=False, na=False, regex=True)
    opinion_match = filtered_df['review_text'].astype(str).str.lower().str.contains('|'.join(opinion_words), case=False, na=False, regex=True)
    #matching rows with the filtered DF after all the rating and other checks have been finished
    matching_rows_and = filtered_df[aspect_match & opinion_match]
    matching_rows_or = filtered_df[aspect_match | opinion_match]
    #just printing rows
    print("\nRows matching both aspect and opinion with correct rating, high helpful count, verified purchase:")
    if len(matching_rows_and) > 0:
        for review_id in matching_rows_and['review_id']:
            print(str(review_id).strip("'"))
    else:
        print("No matches found")
    #anywhere this strip is used I did debug with AI since i could no figure out the sytnax to remove the '' around the review ids

    return matching_rows_and, matching_rows_or


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load data + resources
    file_path = os.path.join(script_dir, "reviews_segment.pkl")
    df = load_data_from_pkl(file_path, force_update=False)

    positive_file = os.path.join(script_dir, "positive-words.txt")
    negative_file = os.path.join(script_dir, "negative-words.txt")

    positive_words = load_word_list(positive_file)
    negative_words = load_word_list(negative_file)

    queries = [
        "audio quality: poor",
        "wifi signal: strong",
        "mouse button: click problem",
        "gps map: useful",
        "image quality: sharp"
    ]

    outputs_dir = os.path.join(script_dir, "Outputs")
    advanced_folder = os.path.join(outputs_dir, "AdvancedModel")

    os.makedirs(advanced_folder, exist_ok=True)

    for query in queries:
        aspect = query.split(":")[0].strip().lower().replace(" ", "_")

        # -------------------------------
        # Test 4 Output (Rating Search)
        # -------------------------------
        output_file = os.path.join(advanced_folder, f"{aspect}_test5.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("#" * 80)
            print("# M1 : Rating-Based Search")
            print("#" * 80)
            print(f"QUERY: {query}")
            
            matching_rows_and, matching_rows_or = ratingSearch(query, df, positive_words, negative_words)

            sys.stdout = original_stdout

        print(f"Created output file: {output_file}")


if __name__ == "__main__":
    main()