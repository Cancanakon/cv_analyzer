from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from docx import Document
import spacy
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import spacy


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def download_and_load_model(model_name):
    import subprocess
    import sys

    try:
        subprocess.check_output([sys.executable, "-m", "spacy", "download", model_name])
    except subprocess.CalledProcessError as e:
        print("Error occurred while downloading and installing the model:", e)
        sys.exit(1)

download_and_load_model("en_core_web_lg")

nlp = spacy.load("en_core_web_lg")
def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_model():
    with open('model/cv_analyzer_model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

def analyze_cv(cv_text, required_skills):
    model, vectorizer = load_model()
    cv_text = preprocess_text(cv_text)
    vectorized_text = vectorizer.transform([cv_text])

    analysis = {
        'matching_skills': {},
        'missing_skills': [],
        'percentage_match': 0
    }

    if required_skills:
        for skill in required_skills:
            if skill.lower() in cv_text:
                analysis['matching_skills'][skill] = cv_text.count(skill.lower())
            else:
                analysis['missing_skills'].append(skill)

        if len(required_skills) > 0:
            analysis['percentage_match'] = (len(analysis['matching_skills']) / len(required_skills)) * 100

    return analysis

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def init_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            surname TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            cv_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def extract_information_from_cv(cv_text):
    doc = nlp(cv_text)

    name = None
    surname = None
    address = None
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            name_parts = ent.text.split()
            if len(name_parts) > 1:
                name = name_parts[0]
                surname = name_parts[-1]
                break
        if ent.label_ == 'GPE' or ent.label_ == 'LOC':
            address = ent.text

    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    email_match = email_pattern.search(cv_text)
    email = email_match.group(0) if email_match else None

    phone_pattern = re.compile(r'[(]?(\d{3})[)]?[\s-]?(\d{3})[\s-]?(\d{4})')
    phone_match = phone_pattern.search(cv_text)
    phone = " ".join(phone_match.groups()) if phone_match else None

    return name, surname, email, phone, address

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        required_skills = request.form['skills'].split(',')
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.rsplit('.', 1)[1].lower() == 'pdf':
                cv_text = extract_text_from_pdf(filepath)
            elif filename.rsplit('.', 1)[1].lower() == 'docx':
                cv_text = extract_text_from_docx(filepath)

            name, surname, email, phone, address = extract_information_from_cv(cv_text)

            name = name if name else "Unknown"
            surname = surname if surname else "Unknown"
            email = email if email else "Unknown"
            phone = phone if phone else "Unknown"
            address = address if address else "Unknown"

            analysis = analyze_cv(cv_text, required_skills)

            return render_template('result.html', analysis=analysis, name=name, surname=surname, email=email, phone=phone, address=address, cv_text=cv_text)

    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    name = request.form['name']
    surname = request.form['surname']
    email = request.form['email']
    phone = request.form['phone']
    address = request.form['address']
    cv_text = request.form['cv_text']

    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO cv_data (name, surname, phone, email, address, cv_text)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, surname, phone, email, address, cv_text))
    conn.commit()
    conn.close()

    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    app.run(debug=True)
