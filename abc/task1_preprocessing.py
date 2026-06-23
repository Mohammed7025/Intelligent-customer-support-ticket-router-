import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # 1a) Create baseline dataset containing at least 6 mock customer service tickets.
    # Columns: ticketid, textcontent, groundtruthdept (tech, billing and account).
    data = {
        'ticketid': [1, 2, 3, 4, 5, 6, 7],
        'textcontent': [
            "My internet is not working since yesterday!! Error code 404.",
            "I want to change my password but the link is broken.",
            "Why was I charged $50.99 this month? This is incorrect.",
            "Please cancel my subscription immediately! Account #12345.",
            "I cannot access my account, it says user not found.",
            "The router is blinking red and I have no WiFi signal.",
            "Can I get a copy of last month's invoice sent to my email?"
        ],
        'groundtruthdept': ['tech', 'tech', 'billing', 'account', 'account', 'tech', 'billing']
    }

    df = pd.DataFrame(data)
    print("--- 1a) Original Dataset ---")
    print(df[['ticketid', 'textcontent', 'groundtruthdept']])
    print("\n")

    # 1b) Implement text preprocessing function
    def preprocess_text(text):
        # 1b i) convert all text to lowercase
        text = text.lower()
        # 1b ii) remove punctuation, special char & numerical digits
        # We only keep letters a-z and whitespace.
        text = re.sub(r'[^a-z\s]', ' ', text)
        # 1b iii) strips extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Apply preprocessing to the textcontent column
    df['cleaned_text'] = df['textcontent'].apply(preprocess_text)
    
    print("--- 1b) Cleaned Dataset ---")
    print(df[['ticketid', 'cleaned_text']])
    print("\n")

    # 1c) Use TfidfVectorizer to transform the cleaned text into a numerical feature matrix
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df['cleaned_text'])

    print("--- 1c) TF-IDF Feature Matrix ---")
    print("Vocabulary Array:")
    print(vectorizer.get_feature_names_out())
    print("\nFeature Matrix Shape:", feature_matrix.shape)
    print("\nDense Matrix Representation:")
    print(feature_matrix.toarray())

if __name__ == "__main__":
    main()
