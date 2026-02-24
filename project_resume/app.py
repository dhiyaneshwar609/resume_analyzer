import re

from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


import pdfplumber

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def calculate_similarity(resume_text, job_desc):
    documents = [resume_text, job_desc]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return round(similarity[0][0] * 100, 2)



def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    score = None

    if request.method == "POST":
        resume_file = request.files.get("resume")
        job_desc = request.form.get("job_desc", "")

        resume_text = extract_text(resume_file) if resume_file else ""

        print("[DEBUG] resume_text length:", len(resume_text))
        print("[DEBUG] job_desc length:", len(job_desc))

        if not resume_text.strip() or not job_desc.strip():
            score = 0.0
        else:
            score = calculate_similarity(resume_text, job_desc)

    return render_template("index.html", score=score)

if __name__ == "__main__":
    app.run(debug=True)
