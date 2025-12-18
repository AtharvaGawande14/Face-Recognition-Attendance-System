import cv2
import numpy as np
import os
from datetime import datetime
import csv

# Load the trained face recognizer and Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Example names (adjust IDs according to your training)
names = ['None', 'Atharva Gawande', 'Rohit Halmare', 'Namit Meshram', 'Manas Gupta', 'Saptak Kashi', 'Harshal kanojiya']  

# Track already marked attendance for the session
attendance_marked = set()

# Function to mark attendance

def mark_attendance(name):
    # Ensure attendance folder exists
    if not os.path.exists("Attendance"):
        os.makedirs("Attendance")

    # File name based on date
    filename = f"Attendance/attendance_{datetime.now().strftime('%Y_%m_%d')}.csv"

    # Create the file with headers if it doesn't exist
    if not os.path.isfile(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    #  Only mark once per session & ignore "Unknown"
    if name not in attendance_marked and name != "Unknown":
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])

        attendance_marked.add(name)  # Add to set so it won't repeat



# Initialize and start the video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 80:
            name = names[id]
            mark_attendance(name)
            color = (0, 255, 0)   # green
            status = "Present"
        else:
            name = "Unknown"
            color = (0, 0, 255)   # red
            status = "Unknown"

        # Draw box
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Name above the face
        cv2.putText(img, name, (x, y-10), font, 1, color, 2)

        # Status below the face
        cv2.putText(img, status, (x, y+h+30), font, 1, color, 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # 'ESC' to exit
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
