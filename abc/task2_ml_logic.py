import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Setup data from Task 1
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
    df['cleaned_text'] = df['textcontent'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df['cleaned_text'])

    # 2a) Define reference vector for each dept by averaging sample preprocessed ticket vectors
    print("--- 2a) Creating Reference Vectors ---")
    print("Method used: Averaging preprocessed ticket TF-IDF vectors (Centroid-based classification).")
    
    departments = df['groundtruthdept'].unique()
    reference_vectors = {}
    
    for dept in departments:
        # Get indices of tickets for this department
        indices = df[df['groundtruthdept'] == dept].index
        # Extract corresponding vectors from the sparse TF-IDF matrix
        dept_vectors = feature_matrix[indices]
        # Average across rows (axis=0) to create a single centroid vector for the department
        dept_centroid = np.asarray(dept_vectors.mean(axis=0))
        reference_vectors[dept] = dept_centroid
        
    print(f"Calculated reference vectors for classes: {list(reference_vectors.keys())}\n")

    # Let's test with new incoming tickets
    incoming_tickets = [
        "How do I reset my router to get internet?", # Tech-related
        "My bill is too high this month. I was charged extra.", # Billing-related
        "Where is the company cafeteria?", # Out-of-domain (should fall below threshold)
        "I need to cancel my subscription and close my account." # Account-related
    ]
    
    incoming_cleaned = [preprocess_text(t) for t in incoming_tickets]
    incoming_matrix = vectorizer.transform(incoming_cleaned)

    print("--- 2b & 2c) Cosine Similarity and Confidence Thresholding ---")
    print("Evaluating new incoming tickets...\n")
    
    for i, ticket_text in enumerate(incoming_tickets):
        vector = incoming_matrix[i].toarray()
        
        # 2b) Calculate cosine similarity alignment with target classes
        similarities = {}
        for dept, ref_vec in reference_vectors.items():
            # cosine_similarity expects 2D arrays, returns a 2D distance matrix
            sim = cosine_similarity(vector, ref_vec)[0][0]
            similarities[dept] = sim
            
        # Find the max similarity score
        predicted_dept = max(similarities, key=similarities.get)
        highest_score = similarities[predicted_dept]
        
        print(f"Ticket: '{ticket_text}'")
        for dept, score in similarities.items():
            print(f"  - {dept.capitalize()} Similarity: {score:.4f}")
            
        # 2c) Flag uncertain tickets below 0.5 threshold
        if highest_score < 0.5:
            print(f"-> Prediction: UNCERTAIN (Highest score {highest_score:.4f} for {predicted_dept.upper()} is below 0.5 threshold)")
        else:
            print(f"-> Prediction: {predicted_dept.upper()} (Score {highest_score:.4f} >= 0.5)")
        print("-" * 50)

if __name__ == "__main__":
    main()
