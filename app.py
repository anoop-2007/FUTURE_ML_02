"""
app.py — Support Ticket Classification & Prioritization (Live Demo)

A Streamlit app that loads pre-trained models and predicts ticket
category + priority from user-entered text.
"""

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ---- Page setup ----
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# ---- Load models (cached so this only runs once, not on every interaction) ----
@st.cache_resource
def load_models():
    vectorizer = joblib.load('vectorizer.pkl')
    category_model = joblib.load('category_model.pkl')
    priority_model = joblib.load('priority_model.pkl')
    return vectorizer, category_model, priority_model

vectorizer, category_model, priority_model = load_models()

# ---- Same text cleaning logic used in training ----
stop_words = set(stopwords.words('english'))
negations = {'not', 'no', "n't", 'never', "don't", "doesn't", "isn't", "wasn't"}
stop_words = stop_words - negations

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ---- UI ----
st.title("🎫 Support Ticket Classifier")
st.write(
    "Paste a customer support ticket below to automatically predict its "
    "**category** and **priority level**."
)

ticket_text = st.text_area(
    "Ticket description",
    placeholder="e.g. My device is not turning on and I've already tried restarting it twice...",
    height=140
)

if st.button("Classify Ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter some ticket text first.")
    else:
        cleaned = clean_text(ticket_text)
        vec = vectorizer.transform([cleaned])

        category_pred = category_model.predict(vec)[0]
        priority_pred = priority_model.predict(vec)[0]

        category_proba = category_model.predict_proba(vec)[0].max()
        priority_proba = priority_model.predict_proba(vec)[0].max()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Category", category_pred)
            st.caption(f"Confidence: {category_proba:.0%}")
        with col2:
            priority_colors = {
                "Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"
            }
            icon = priority_colors.get(priority_pred, "")
            st.metric("Predicted Priority", f"{icon} {priority_pred}")
            st.caption(f"Confidence: {priority_proba:.0%}")

        st.divider()
        st.caption(
            "⚠️ This model predicts based on ticket text only. Real-world "
            "priority decisions may also depend on customer tier, history, "
            "and business context not captured here."
        )

st.divider()
with st.expander("About this project"):
    st.markdown("""
    This is a live demo of a machine learning system for
    **Support Ticket Classification & Prioritization**.

    - **Approach:** TF-IDF vectorization + Logistic Regression
    - **Category model:** predicts Billing / Technical Issue / Refund Request /
      Cancellation Request / Product Inquiry
    - **Priority model:** predicts Low / Medium / High / Critical

    Full training notebook, evaluation metrics, and methodology are available
    in the GitHub repository.
    """)
