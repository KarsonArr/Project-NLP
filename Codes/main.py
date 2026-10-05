import pandas as pd
import os
import sys
from dataXL import load_data_from_pkl
import spacy
from spacy.matcher import PhraseMatcher
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


#boolean search on the review column since that is all that my target is
def boolSearch(query, aspect, opinion, df):
    aspect_words = aspect.split()
    opinion_words = opinion.split()
    filtered_df = df['review_text'].astype(str).str.lower()
    aspect_match = filtered_df.str.contains('|'.join(aspect_words), case=False, na=False, regex=True)
    opinion_match = filtered_df.str.contains('|'.join(opinion_words), case=False, na=False, regex=True)
    matching_aspect = df[aspect_match]
    matching_rows_and = df[aspect_match & opinion_match]
    matching_rows_or = df[aspect_match | opinion_match]

    print("Rows matching both aspect and opinion:")
    if len(matching_rows_and) > 0:
        for review_id in matching_rows_and['review_id']:
            print(str(review_id).strip("'"))

    print("\nRows matching either aspect or opinion:")
    if len(matching_rows_or) > 0:
        
        for review_id in matching_rows_or['review_id']:
            print(str(review_id).strip("'"))

    return matching_aspect, matching_rows_and, matching_rows_or


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

#need to better match aspects with relevant words
def matching_machine(nlp, aspect):
    #using this to better tokenize phrases to find more accurate matches with the query
    #uses phrase matcher from spacy
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    patterns = [nlp(aspect)] 
    matcher.add("ASPECT", patterns)

    return matcher
def parse_aspect_tree(df, query, nlp, positive_words, negative_words):
    #need to get all documents with the aspect, also getting opinion
    parts = query.split(":")
    aspect = parts[0].strip().lower()
    opinion = parts[1].strip().lower()
    aspect_words = aspect.split()
    opinion_words = opinion.split()
    aspect_match = df['review_text'].astype(str).str.lower().str.contains(aspect, case=False, na=False)
    matching_aspect = df[aspect_match]
    #i also need to check for similar opinionated words, so like good = great sharp etc
    target_opinions = set(opinion_words)
    if any(w in positive_words for w in opinion_words):
        target_opinions.update(positive_words)
    elif any(w in negative_words for w in opinion_words):
        target_opinions.update(negative_words)
    #now need to make a parse tree for each document with the aspect, checking to see if opinion is nearby in the tree
    seen_indexes = set()
    #looking at every sentence from matching aspect reviews
    #used AI here to help some since i wasnt getting any result
    target_opinions_lemmas = {t.lower() for t in target_opinions}
    filtered_df = []
    seen_indexes = set()
    #distance, setting how far the tree is allowed to look
    MAX_DISTANCE = 5
    #needed to better match so calling the function
    matcher = matching_machine(nlp, aspect)
    for index, row in matching_aspect.iterrows():
        # doing this to prevent duplicates
        if index in seen_indexes:
            continue

        doc = nlp(row['review_text'])
        doc_found = False
        matches = matcher(doc)
        for match_id,start,end in matches:
            if doc_found: break
            # fixed to handle bigger aspects
            #making span
            span = doc[start:end]
            #this is the particular sentence what it has found
            # looking deeper into the tree at the aspects
            # Check the subtree
            for tok in span:
                found_via_tree = False
            #going down in the tree
                for child in tok.subtree:
                    if abs(child.i - tok.i) <= MAX_DISTANCE:
                        if child.lemma_.lower() in target_opinions_lemmas:
                            if child.dep_ in ("amod", "acomp", "attr", "oprd", "dobj", "prep", "pobj", "advmod", "ROOT"): 
                                found_via_tree = True
                                break
            # checking the parents, going up the tree
            if not found_via_tree:
                    for ancestor in tok.ancestors:
                        if abs(ancestor.i - tok.i) <= MAX_DISTANCE:
                            if ancestor.lemma_.lower() in target_opinions_lemmas:
                                if ancestor.pos_ in ("ADJ", "VERB", "NOUN"): 
                                    found_via_tree = True
                                    break
            #checking the siblings, AKA horizontally around the word it is looking at
            if not found_via_tree and tok.head:
                    for sibling in tok.head.children:
                        if sibling != tok:
                            if abs(sibling.i - tok.i) <= MAX_DISTANCE:
                                if sibling.lemma_.lower() in target_opinions_lemmas:
                                    if sibling.pos_ in ("ADJ", "VERB", "ADV", "NOUN"):
                                        found_via_tree = True
                                        break      
            #adding found ones.
            if found_via_tree:
                filtered_df.append(row)
                seen_indexes.add(index)
                doc_found = True
                break
                #now i had the issue from this that my tree was barely getting any results.
    #just outputting the filtered df now      
    filtered_df = pd.DataFrame(filtered_df)
    filtered_df = filtered_df[filtered_df['amazon_verified_purchase'] == '1']
    print(f"\nRows matching aspect '{aspect}' and opinion '{opinion}' in parse tree:")
    if len(filtered_df) > 0:
        for review_id in filtered_df['review_id']:
            print(str(review_id).strip("'"))
    else:
        print("No matches found")
    
    return filtered_df



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load data + resources
    file_path = os.path.join(script_dir, "reviews_segment.pkl")
    df = load_data_from_pkl(file_path, force_update=False)

    positive_file = os.path.join(script_dir, "positive-words.txt")
    negative_file = os.path.join(script_dir, "negative-words.txt")

    nlp = spacy.load("en_core_web_sm")
    positive_words = load_word_list(positive_file)
    negative_words = load_word_list(negative_file)

    queries = [
        "audio quality: poor",
        "wifi signal: strong",
        "mouse button: click problem",
        "gps map: useful",
        "image quality: sharp"
    ]

    # -------------------------------
    # FIXED FOLDER STRUCTURE
    # -------------------------------
    outputs_dir = os.path.join(os.path.dirname(script_dir), "Outputs")
    baseline_folder = os.path.join(outputs_dir, "Baseline")
    advanced_folder = os.path.join(outputs_dir, "AdvancedModel")

    os.makedirs(baseline_folder, exist_ok=True)
    os.makedirs(advanced_folder, exist_ok=True)


    #the way this outputs was mostly generated by AI since it isnt relevant to the project
    for query in queries:
        aspect = query.split(":")[0].strip().lower().replace(" ", "_")

        parts = query.split(":")
        aspect_term = parts[0].strip().lower()
        opinion_term = parts[1].strip().lower()

        aspect_only, aspect_and_opinion, aspect_or_opinion = boolSearch(
            query, aspect_term, opinion_term, df
        )

        # -------------------------------
        # Test 1 Output
        # -------------------------------
        output_file = os.path.join(baseline_folder, f"{aspect}_test1.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("#" * 80)
            print("# Test 1: Boolean Aspect Term Retrieval")
            print("#" * 80)
            print(f"QUERY: {query}")
            print("\nReviews containing aspect term:")
            if len(aspect_only) > 0:
                for review_id in aspect_only['review_id'].values:
                    print(str(review_id).strip("'"))
            else:
                print("No matches found")

            sys.stdout = original_stdout

        # -------------------------------
        # Test 2 Output
        # -------------------------------
        output_file = os.path.join(baseline_folder, f"{aspect}_test2.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("#" * 80)
            print("# Test 2: Aspect AND Opinion Match")
            print("#" * 80)
            print(f"QUERY: {query}")
            print("\nReviews containing both aspect and opinion:")
            if len(aspect_and_opinion) > 0:
                for review_id in aspect_and_opinion['review_id'].values:
                    print(str(review_id).strip("'"))
            else:
                print("No matches found")

            sys.stdout = original_stdout

        # -------------------------------
        # Test 3 Output
        # -------------------------------
        output_file = os.path.join(baseline_folder, f"{aspect}_test3.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("#" * 80)
            print("# Test 3: Aspect OR Opinion Match")
            print("#" * 80)
            print(f"QUERY: {query}")
            print("\nReviews containing either aspect or opinion:")
            if len(aspect_or_opinion) > 0:
                for review_id in aspect_or_opinion['review_id'].values:
                    print(str(review_id).strip("'"))
            else:
                print("No matches found")

            sys.stdout = original_stdout
        # -------------------------------
        # Test 4 Output ( Parse Tree)
        # -------------------------------
        output_file = os.path.join(advanced_folder, f"{aspect}_test4.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("#" * 80)
            print("# M2: Parse tree based search results")
            print("#" * 80)
            parse_aspect_tree(df, query, nlp, positive_words, negative_words)

            sys.stdout = original_stdout
if __name__ == "__main__":
    main()