import cv2
import os
import time
import dlib
from flask import redirect, url_for

# Initialize camera globally only once
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)
    return faces
        
def draw_rectangle(faces, frame):
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def gen_frames():
    while True:
        if cap.isOpened():  # Check if the camera is available
            ret, frame = cap.read()
            if not ret:
                break
            else:
                faces = detect_faces(frame)
                draw_rectangle(faces, frame)

                flipped_frame = cv2.flip(frame, 0)
                yield flipped_frame
        else:
            break


def release_camera():
    if cap.isOpened():
        cap.release()
        
def take_pictures(user_id):
    # Ensure the folder exists
    user_folder = os.path.join('dataset', user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    print("Opening the camera...")
    if not cap.isOpened():
        cap.open(0)

    pictures_taken = 0
    while pictures_taken < 100:
        ret, frame = cap.read()
        if ret:
            faces = detect_faces(frame)
            if len(faces) > 0:  # Take picture only if a face is detected
                file_path = os.path.join(user_folder, f'{pictures_taken + 1}.jpg')
                cv2.imwrite(file_path, frame)
                pictures_taken += 1
                print(f"Picture {pictures_taken} taken.")
        time.sleep(0.5)

    print("Closing the camera...")
    release_camera()
    
def take_auth_pic():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # Capture a new frame
    if not ret:
        print("Failed to grab frame.")
        return None
            
    faces = detect_faces(frame)  # Detect faces in the current frame
    if len(faces) == 1: 
        print("Face Detected") # Take picture only if exactly one face is detected
        cv2.imwrite('auth.jpg', frame)  # Save the captured frame as 'auth.jpg'
        return 'auth.jpg'  # Return the path to the saved image
    else:
        print("No valid face detected.")
        return None