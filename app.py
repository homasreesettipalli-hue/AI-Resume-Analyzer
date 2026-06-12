import streamlit as st
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
import PyPDF2
from skills import SKILLS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(pdf_file):
    text = ""

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text.lower()

def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return found_skills

st.title("AI Resume Analyzer")

resume = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if resume and job_description:

    resume_text = extract_text(resume)

    resume_skills = extract_skills(resume_text)

    jd_skills = extract_skills(job_description.lower())

    vectors = CountVectorizer().fit_transform(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(vectors)[0][1]

    score = round(similarity * 100)

    missing_skills = list(
        set(jd_skills) - set(resume_skills)
    )

    st.subheader("ATS Match Score")

    st.metric(
    label="Resume Score",
    value=f"{score}%"
)

    st.progress(score)
    if score >= 80:
        st.success("Excellent match for this role!")
    elif score >= 60:
        st.warning("Good match, but improve a few skills.")
    else:
        st.error("Low match. Consider adding the missing skills.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Skills Found")
        st.write(resume_skills)

    with col2:
        st.subheader("❌ Missing Skills")
        st.write(missing_skills)

    st.subheader("Suggestions")

    for skill in missing_skills:
        st.write(f"Learn {skill}")