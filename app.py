# app.py - Flask application for AI-powered resume screening
from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx2txt
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "resume_screening_secret_key"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load NLP model
nlp = spacy.load("en_core_web_md")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

def extract_skills(text):
    doc = nlp(text)
    skills = []
    # This is a simplified version - in a real app, you'd have a comprehensive skills database
    skill_keywords = ["python", "java", "javascript", "html", "css", "react", "angular", 
                      "node.js", "sql", "mongodb", "aws", "docker", "kubernetes", 
                      "machine learning", "data analysis", "project management", 
                      "leadership", "communication", "teamwork", "problem solving"]
    
    for skill in skill_keywords:
        if skill.lower() in text.lower():
            skills.append(skill)
    return skills

def score_resume(resume_text, job_description):
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_description, resume_text])
    
    # Calculate similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity * 100  # Convert to percentage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'job_description' not in request.form or not request.form['job_description'].strip():
        flash('No job description provided!', 'error')
        return redirect(request.url)
    
    job_description = request.form['job_description']
    session['job_description'] = job_description
    
    if 'resumes' not in request.files:
        flash('No resume files selected!', 'error')
        return redirect(request.url)
    
    files = request.files.getlist('resumes')
    
    if len(files) == 1 and files[0].filename == '':
        flash('No selected files!', 'error')
        return redirect(request.url)
    
    resume_data = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text
            resume_text = extract_text(file_path)
            if not resume_text:
                flash(f'Could not extract text from {filename}', 'warning')
                continue
            
            # Extract skills
            skills = extract_skills(resume_text)
            
            # Score resume against job description
            match_score = score_resume(resume_text, job_description)
            
            resume_data.append({
                'filename': filename,
                'skills': skills,
                'match_score': match_score
            })
    
    if not resume_data:
        flash('No valid resumes were processed!', 'error')
        return redirect(request.url)
    
    # Sort by match score
    resume_data.sort(key=lambda x: x['match_score'], reverse=True)
    session['resume_data'] = resume_data
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    if 'resume_data' not in session or 'job_description' not in session:
        flash('No data to display. Please upload resumes first.', 'error')
        return redirect(url_for('index'))
    
    resume_data = session['resume_data']
    job_description = session['job_description']
    
    return render_template('results.html', 
                          resume_data=resume_data,
                          job_description=job_description)

@app.route('/view_resume/<filename>')
def view_resume(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash('File not found!', 'error')
        return redirect(url_for('results'))
    
    resume_text = extract_text(file_path)
    skills = extract_skills(resume_text)
    
    return render_template('view_resume.html',
                          filename=filename,
                          resume_text=resume_text,
                          skills=skills)

if __name__ == '__main__':
    app.run(debug=True)