
import streamlit as st
import cohere
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURATION ---
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]  # ✅ Will work on Streamlit Cloud
co = cohere.Client(COHERE_API_KEY)

USECASE_FILE = "usecases.json"
EMBED_MODEL = "embed-english-v3.0"
THRESHOLD = 0.70  # Semantic similarity threshold

@st.cache_resource
def load_usecases():
    with open(USECASE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers

@st.cache_resource
def embed_questions(questions):
    embeddings = co.embed(texts=questions, model=EMBED_MODEL, input_type="search_document").embeddings
    return np.array(embeddings)

def get_best_match(user_query, questions, answers, question_embeddings):
    user_embedding = co.embed(texts=[user_query], model=EMBED_MODEL, input_type="search_query").embeddings[0]
    sims = cosine_similarity([user_embedding], question_embeddings)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= THRESHOLD:
        return answers[best_idx]
    return None  # Fallback if no strong match

def generate_response_fallback(user_query):
    response = co.chat(model="command-r-plus", message=user_query)
    return response.text

def main():
    st.set_page_config(page_title="Adi's KnowAnything", layout="wide")
    st.title("🛠️ Adi's KnowAnything: Knowledge sharing Bot")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask me something about yourself,world,tech,...")

    if user_input:
        questions, answers = load_usecases()
        question_embeddings = embed_questions(questions)

        matched_answer = get_best_match(user_input, questions, answers, question_embeddings)

        if matched_answer:
            response = matched_answer
        else:
            response = generate_response_fallback(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("InfraFix", response))

    for sender, message in st.session_state.history:
        with st.chat_message(sender):
            st.markdown(message)

if __name__ == "__main__":
    main()
