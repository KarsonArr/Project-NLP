Python Packages Required:
--------------------------
- pandas
- spacy
- os (built-in)
- sys (built-in)

Installation Commands:
----------------------
pip install pandas spacy

SpaCy Language Model:
---------------------
python -m spacy download en_core_web_sm

REQUIRED FILES
this file needs to be run first to clean the data.
dataXL.py
this is the data that will be cleaned
reviews_segment.pkl
these two are needed as txt files
positive-words.txt
negative-words.txt
this is where M2 and the other three tests will output from
main.py
this is the file that will output M1 if needed.
m1.py

The output folder has two folders. 
Baseline: Contains the three outputs for the boolean tests.
Advanced: Will contain the output from my M2 model, and if M1.py is ran it will also output to there as test5