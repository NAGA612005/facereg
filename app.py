# app.py
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Initialize DB
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (name TEXT, date TEXT, time TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
registered_name = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global registered_name
    if request.method == 'POST':
        registered_name = request.form['name']
        return redirect(url_for('register_camera'))
    return render_template('register.html')

@app.route('/register_camera')
def register_camera():
    return Response(register_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def register_frames():
    cap = cv2.VideoCapture(0)
    saved = False
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if not saved:
                os.makedirs('faces', exist_ok=True)
                face_img = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
                cv2.imwrite(f"faces/{registered_name}.jpg", face_img)
                saved = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{registered_name} Registered", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/video_feed')
def video_feed():
    return Response(attendance_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def attendance_frames():
    known_faces = {}
    for file in os.listdir('faces'):
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join('faces', file))
            img = cv2.resize(img, (100, 100))
            known_faces[os.path.splitext(file)[0]] = img

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
            name = "Unknown"
            for known_name, known_img in known_faces.items():
                diff = cv2.absdiff(known_img, face_img)
                if diff.mean() < 40:
                    name = known_name
                    now = datetime.now()
                    today = now.date().isoformat()
                    conn = sqlite3.connect('attendance.db')
                    c = conn.cursor()
                    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
                    if not c.fetchone():
                        c.execute("INSERT INTO attendance VALUES (?, ?, ?)", (name, today, now.strftime("%H:%M:%S")))
                        conn.commit()
                    conn.close()
                    break

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/attendance_records')
def attendance_records():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    rows = c.fetchall()
    conn.close()
    return render_template('attendance_records.html', attendance=rows)

if __name__ == '__main__':
    app.run(debug=True)
