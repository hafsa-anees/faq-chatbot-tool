import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faqs import FAQS

# ---------- NLTK data (agar pehle se hai to dobara download nahi hoga) ----------
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------- Text saaf karne wala function ----------
def preprocess(text):
    text = text.lower()                                   # chhote alfaaz
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation hatao
    tokens = word_tokenize(text)                          # alfaaz alag alag karo
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens
              if w not in STOP_WORDS]                     # faltu alfaaz hatao
    return " ".join(tokens)

# ---------- FAQs ko tayar karo (sirf ek baar) ----------
@st.cache_resource
def build_model():
    questions = [faq["question"] for faq in FAQS]
    cleaned = [preprocess(q) for q in questions]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix

vectorizer, faq_matrix = build_model()

# ---------- Best jawab dhoondo ----------
def get_answer(user_question, threshold=0.2):
    cleaned = preprocess(user_question)
    if not cleaned.strip():
        return "Please type a valid question.", 0.0

    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, faq_matrix)[0]
    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < threshold:
        return ("Sorry, I could not understand your question. "
                "Please try asking in a different way."), best_score

    return FAQS[best_index]["answer"], best_score

# ---------- Chat UI ----------
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 Online Shop FAQ Chatbot")
st.caption("Ask me about delivery, returns, payments, orders and more.")

# Sidebar mein FAQs ki list
with st.sidebar:
    st.header("Sample Questions")
    for faq in FAQS:
        st.write("•", faq["question"])

# Chat history yaad rakhna
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! 👋 How can I help you today?"}
    ]

# Purani chat dikhao
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Naya sawal
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, score = get_answer(user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Match score: {score:.2f}")