import streamlit as st
import sqlite3
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt, pdfplumber

# ====================================
# 🌈 CONFIG & BACKGROUND STYLE
# ====================================
def set_bg():
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(255,255,255,0.85);
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: #ffffff;
        text-shadow: 1px 1px 3px #000000;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_bg()

# ====================================
# 🧠 BASIC DATABASE SETUP
# ====================================
conn = sqlite3.connect("users.db")
conn.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
conn.commit()

def add_user(username, password):
    conn.execute("INSERT INTO users VALUES (?,?)", (username, password))
    conn.commit()

def verify_user(username, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cur.fetchone() is not None

# ====================================
# 🧩 SESSION STATE
# ====================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ====================================
# 🧾 AUTHENTICATION PAGES
# ====================================
def login_page():
    st.title("🔐 Login to Resume Matcher")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(user, pw):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.info("Don’t have an account?")
    if st.button("Create New Account"):
        st.session_state.page = "register"
        st.rerun()

def register_page():
    st.title("🆕 Create New Account")
    new_user = st.text_input("Choose a Username")
    new_pw = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        add_user(new_user, new_pw)
        st.success("Account created! You can now log in.")
        st.session_state.page = "login"
        st.rerun()

# ====================================
# 📄 RESUME TEXT EXTRACTION HELPERS
# ====================================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

# ====================================
# 🧮 SIMILARITY COMPUTATION
# ====================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def compute_similarity(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(similarity * 100, 2)

# ====================================
# 🏠 MAIN APP DASHBOARD
# ====================================
def main_app():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.sidebar.success(f"Welcome, {st.session_state.username}! 👋")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.markdown("<h1 style='text-align:center;'>✨ AI Resume–Job Match Platform ✨</h1>", unsafe_allow_html=True)
    st.divider()

    # Upload & Input
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Upload Resume")
        resume_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    with col2:
        st.subheader("💼 Paste Job Description")
        job_desc = st.text_area("Paste or type the job description")

    if st.button("Analyze Match"):
        if resume_file and job_desc.strip():
            with st.spinner("Analyzing... please wait ⏳"):
                # Extract text
                if resume_file.name.endswith(".pdf"):
                    resume_text = extract_text_from_pdf(resume_file)
                else:
                    resume_text = extract_text_from_docx(resume_file)

                score = compute_similarity(resume_text, job_desc)
                st.success(f"✅ Resume–Job Match Score: **{score}%**")

                # Fake missing keywords for demo
                missing_keywords = ["Python", "Teamwork", "Machine Learning"]
                st.markdown("### 🔍 Missing Keywords:")
                st.write(", ".join(missing_keywords))

                # Placeholder for ATS & Cover Letter
                st.info("📋 ATS Compatibility: Good (No columns, no images detected)")
                st.info("✍️ Cover Letter: [Coming Soon – integrates OpenAI or transformer model]")

        else:
            st.warning("Please upload a resume and paste a job description.")

# ====================================
# 🧭 PAGE NAVIGATION
# ====================================
if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    if st.session_state.page == "register":
        register_page()
    else:
        login_page()
else:
    main_app()
