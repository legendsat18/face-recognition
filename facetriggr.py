import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import Tk, Button
from threading import Thread

# Initialize global variables
cap = None
running = False

# Load training images
path = 'Students'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


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


# Encoding training images
encodeListKnown = findEncodings(images)
print('Encoding Complete')


def start_recognition():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True

    while running:
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


def stop_recognition():
    global running
    running = False


# Tkinter GUI
root = Tk()
root.title("Face Recognition System")

start_button = Button(root, text="Start", command=lambda: Thread(target=start_recognition).start(), width=10)
start_button.pack(pady=10)

stop_button = Button(root, text="Stop", command=stop_recognition, width=10)
stop_button.pack(pady=10)

exit_button = Button(root, text="Exit", command=root.quit, width=10)
exit_button.pack(pady=10)

root.mainloop()
