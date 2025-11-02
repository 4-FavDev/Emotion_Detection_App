from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3

app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create / connect to database
conn = sqlite3.connect('emotions.db', check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        image_path TEXT,
        emotion TEXT
    )
""")
conn.commit()

# Load the trained model at startup
model = tf.keras.models.load_model("emotion_model.h5")

# Hardcoded emotion labels
labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    image_file = request.files['image']

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Read and preprocess image correctly for your model
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, (48, 48))                     # Resize to 48x48
    img = img / 255.0                                   # Normalize
    img = np.expand_dims(img, axis=-1)                  # Add channel dimension
    img = np.expand_dims(img, axis=0)                   # Add batch dimension

    # Predict emotion
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    emotion = labels[class_index]

    # Save to database
    cursor.execute(
        "INSERT INTO users (name, image_path, emotion) VALUES (?, ?, ?)",
        (name, image_path, emotion)
    )
    conn.commit()

    # Show result
    return render_template('index.html', result=emotion, img=image_path)

if __name__ == '__main__':
    app.run(debug=True)
