import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load Catalog ----
@st.cache_data
def load_catalog():
    df = pd.read_csv("shl_catalog.csv")
    df.fillna("", inplace=True)
    df["combined"] = df["job_roles"] + " " + df["skills_assessed"]
    return df

catalog = load_catalog()

# ---- Setup Gemini Pro ----
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Set this in .streamlit/secrets.toml
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-pro")


# ---- Extract Intent from Natural Language ----
def extract_intent(user_query):
    prompt = f"""
    You are an AI assistant helping match job roles and assessments.
    Extract the following fields from the user's input:
    - job_role
    - required_skills
    - preferred_difficulty
    - max_duration (optional)

    Input: "{user_query}"

    Respond in this format:
    job_role: <role>
    required_skills: <comma-separated-skills>
    preferred_difficulty: <Low/Medium/High>
    max_duration: <minutes or unknown>
    """
    response = model.generate_content(prompt)
    return response.text

# ---- Match Intent to Catalog ----
def recommend_assessments(job_title, skills, difficulty, top_n=5):
    user_input = job_title + " " + skills
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(catalog['combined'])
    user_vec = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_idx = sim_scores.argsort()[-top_n:][::-1]
    results = catalog.iloc[top_idx].copy()
    results['similarity'] = sim_scores[top_idx]
    if difficulty:
        results = results[results['difficulty'].str.lower() == difficulty.lower()]
    return results[['product_name', 'product_url', 'remote_testing', 'irt_supported', 'duration', 'test_type', 'similarity']]

# ---- Streamlit UI ----
st.title("SHL Assessment Recommendation Engine")
st.write("Enter a job description or query, and we'll suggest the most suitable SHL assessments.")

user_input = st.text_area("Paste your job description or type a query:", height=200)

if st.button("Recommend Assessments") and user_input:
    with st.spinner("Understanding your query and finding matches..."):
        parsed = extract_intent(user_input)
        st.subheader("🔍 Extracted Intent")
        st.code(parsed)

        # Parse intent
        try:
            lines = parsed.strip().split("\n")
            role = lines[0].split(":")[1].strip()
            skills = lines[1].split(":")[1].strip()
            difficulty = lines[2].split(":")[1].strip()
        except:
            st.error("❌ Failed to extract intent. Please rephrase your query.")
            st.stop()

        results = recommend_assessments(role, skills, difficulty)

        st.subheader("📋 Recommended Assessments")
        st.dataframe(results)
