import cv2
import os

# Load Haarcascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Haarcascade file NOT FOUND!")
    exit()

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Camera NOT opening!")
    exit()

# Enter User ID
user_id = input('Enter User ID (1,2,3...): ')

# Create dataset folder
if not os.path.exists('dataset'):
    os.makedirs('dataset')

sample_count = 0
print("Capturing images... Look at the camera.")

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Frame not captured!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        sample_count += 1

        cv2.imwrite(f"dataset/User.{user_id}.{sample_count}.jpg", gray[y:y+h, x:x+w])

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"Samples: {sample_count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Dataset Creator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if sample_count >= 50:
        print("Dataset creation completed.")
        break

cam.release()
cv2.destroyAllWindows()
