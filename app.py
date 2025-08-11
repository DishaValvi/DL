import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
label_encoders = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_excel(filepath) if file.filename.endswith('xlsx') else pd.read_csv(filepath)
        df.to_csv('uploads/data.csv', index=False)
        return render_template('preview.html', tables=[df.head().to_html(classes='data')])
    return "File type not allowed"

@app.route('/train', methods=['GET', 'POST'])
def train():
    global model, label_encoders
    df = pd.read_csv('uploads/data.csv')
    
    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump((model, label_encoders), f)

    return render_template('train.html', accuracy=round(acc*100, 2))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        with open('models/model.pkl', 'rb') as f:
            model, label_encoders = pickle.load(f)
        pred = model.predict([features])[0]
        return render_template('predict.html', prediction='Yes' if pred == 1 else 'No')
    return render_template('predict.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True,host="0.0.0.0",port=10000)
