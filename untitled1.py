from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load training images
path = 'Students'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("No face found in an image.")
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

    if name not in nameList:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        with open('Attendance.csv', 'a') as f:
            f.write(f'\n{name},{dtString}')


def send_email(absent_students):
    # Email credentials
    sender_email = "your_email@example.com"  # Replace with your email
    sender_password = "your_app_password"   # Use App Password generated above
    receiver_email = "receiver_email@example.com"

    # Email content
    subject = "Absent Students Notification"
    body = f"The following students were absent:\n\n" + "\n".join(absent_students)

    # Create email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Encoding training images
encodeListKnown = findEncodings(images)


def start_recognition():
    cap = cv2.VideoCapture(0)
    present_students = set()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture from webcam.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                present_students.add(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Identify absent students
    absent_students = set(classNames) - present_students
    if absent_students:
        send_email(absent_students)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/start')
def start():
    threading.Thread(target=start_recognition).start()
    flash("Recognition started. Press 'q' in the webcam window to stop.")
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)
