import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI
st.set_page_config(page_title="Resume Screening System", layout="wide")
st.title("AI-Powered Resume Screening and Ranking System")
st.markdown("Upload resumes as text files or enter them manually.")

# Input fields
job_description = st.text_area("Enter Job Description:")
num_resumes = st.number_input("Number of Resumes:", min_value=1, value=3)

resumes = {}

st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resume Files (TXT only)", accept_multiple_files=True, type=["txt"])

for file in uploaded_files:
    resumes[file.name] = file.read().decode("utf-8")

st.sidebar.markdown("OR Enter resumes manually below:")

for i in range(num_resumes):
    candidate_name = st.text_input(f"Candidate {i+1} Name:", key=f"name_{i}")
    candidate_resume = st.text_area(f"Candidate {i+1} Resume:", key=f"resume_{i}")
    if candidate_name and candidate_resume:
        resumes[candidate_name] = candidate_resume

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and special characters
    words = text.split()
    stopwords = set(["a", "an", "the", "and", "or", "is", "to", "of", "in", "for", "on", "with", "as", "by", "at", "from", "this", "that", "it", "be", "are", "was", "were", "will", "has", "have", "had", "not", "but", "so", "if", "then", "than", "which", "who", "whom"])
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

if st.button("Rank Candidates") and job_description and resumes:
    # Preprocess resumes and job description
    processed_resumes = {candidate: preprocess(text) for candidate, text in resumes.items()}
    processed_job_desc = preprocess(job_description)

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(list(processed_resumes.values()) + [processed_job_desc])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Ranking candidates based on similarity
    ranked_candidates = sorted(
        [(candidate, similarity_scores[0][idx]) for idx, candidate in enumerate(processed_resumes.keys())],
        key=lambda x: x[1],
        reverse=True
    )

    # Display results
    st.subheader("Ranked Candidates")
    for rank, (candidate, score) in enumerate(ranked_candidates, start=1):
        st.write(f"{rank}. {candidate} - Score: {score:.2f}")
