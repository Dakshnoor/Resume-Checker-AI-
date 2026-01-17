from flask import Flask, render_template, request
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pdfplumber
from docx import Document

app = Flask(__name__)

# ---------------- PREPROCESSING ----------------
def preprocess(text):
    text = text.lower()

    # Keyword normalization
    keyword_map = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "dl": "deep learning",
        "BE": "Bachelor of Engineering",
        "Btech": "BE",
    }

    for k, v in keyword_map.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


# ---------------- FILE TEXT EXTRACTION ----------------
def extract_text(file):
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        return text

    elif filename.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    return ""


# ---------------- SKILL EXTRACTION ----------------
COMMON_SKILLS = [
    "python", "machine learning", "deep learning", "data analysis",
    "natural language processing", "computer vision", "scikit learn",
    "tensorflow", "keras", "pandas", "numpy", "flask", "sql"
]

def extract_skills(text):
    found = set()
    for skill in COMMON_SKILLS:
        if skill in text:
            found.add(skill)
    return found


# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    match_percentage = None

    if request.method == "POST":

        # Job Description
        jd_text = request.form.get("job_text")
        if jd_text and jd_text.strip():
            jd_text = preprocess(jd_text)
        else:
            jd_file = request.files.get("job_desc")
            jd_text = preprocess(extract_text(jd_file)) if jd_file else ""

        # Resume
        resume_file = request.files.get("resume")
        resume_text = preprocess(extract_text(resume_file))

        # -------- TF-IDF SIMILARITY (with n-grams) --------
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        vectors = vectorizer.fit_transform([resume_text, jd_text])
        tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # -------- SKILL MATCH SCORE --------
        jd_skills = extract_skills(jd_text)
        resume_skills = extract_skills(resume_text)

        if jd_skills:
            skill_score = len(jd_skills & resume_skills) / len(jd_skills)
        else:
            skill_score = 0

        # -------- KEYWORD DENSITY BONUS --------
        keyword_bonus = min(len(jd_skills & resume_skills) * 0.05, 0.1)

        # -------- FINAL WEIGHTED SCORE --------
        final_score = (
            (0.6 * tfidf_score) +
            (0.3 * skill_score) +
            (0.1 * keyword_bonus)
        )

        match_percentage = round(final_score * 100, 2)

    return render_template("index.html", match_percentage=match_percentage)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
