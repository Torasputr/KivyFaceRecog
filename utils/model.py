import os
import dlib
import cv2
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from utils.user_manager import get_username_from_csv
from utils.camera import release_camera, take_auth_pic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from flask import url_for, redirect, make_response, render_template


dataset_dir = './dataset'

detector = dlib.get_frontal_face_detector()

facenet_model = FaceNet()

label_encoder = LabelEncoder()

def detect_faces(img_path):
    print(f"Image path: {img_path}")
    faces = []
    frame = cv2.imread(img_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector(gray_frame)
    for detection in detections:
        x, y, w, h = detection.left(), detection.top(), detection.width(), detection.height()
        x, y = abs(x), abs(y)
        count = 0
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f"./check/{count}.jpg", face)

        faces.append(face)
    
    return faces

def preprocess_face(face, target_size=(160, 160)):
    face_resized = cv2.resize(face, target_size)
    return face_resized

def extract_face_embeddings(faces):
    embeddings = []
    
    for face in faces:
        preprocessed_face = preprocess_face(face)
        preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
        
        embedding = facenet_model.embeddings(preprocessed_face)
        embeddings.append(embedding[0])
    
    return embeddings

def prepare_data():
    embeddings = []
    labels = []
    image_count = 0
    face_count = 0
    for user_id in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, user_id)
        if os.path.isdir(user_dir):
            for image_name in os.listdir(user_dir):
                img_path = os.path.join(user_dir, image_name)
                image_count += 1
                print(f"Detecting Faces from {image_name}")
                faces = detect_faces(img_path)
                if faces:
                    face_count += 1
                    print("Extracting Face Embeddings")
                    face_embeddings = extract_face_embeddings(faces)
                    embeddings.extend(face_embeddings)
                    labels.extend([user_id] * len(face_embeddings))
    print(f"{face_count}/{image_count} faces detected")
    accuracy = face_count / image_count * 100
    print(f"Face Detection Accuracy: {accuracy}%")
    return embeddings, labels

def train_model(embeddings, labels):
    encoded_labels = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    print(f"Training Accuracy: {classifier.score(X_train, y_train)}")
    print(f"Test Accuracy: {classifier.score(X_test, y_test)}")
    return classifier

def save_model(model, model_file="dlm.h5"):
    model.save(model_file)
    print("Model Saved o7")

def load_keras_model():
    model_file = 'dlm.h5'
    try:
        model = load_model('dlm.h5')
        print(f"Model loaded from {model_file}")
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")

def save_label_encoder(label_encoder, file_path='label_encoder.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(label_encoder, f)

def create_deep_learning_classifier(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_deep_learning_classifier(embeddings, labels):
    encoded_label = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_label)
    embeddings = np.array(embeddings)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, one_hot_labels, test_size=0.2, random_state=42)
    
    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1]
    classifier = create_deep_learning_classifier(input_shape, num_classes)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    classifier.fit(X_train, y_train,
                   epochs=5,
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   verbose=1)
    
    train_loss, train_accuracy = classifier.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = classifier.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    
    return classifier, label_encoder

def detect_and_extract_embeddings():
    img_path = take_auth_pic()
    if img_path is None:
        print("Failed to capture image for authentication.")
        return None

    print(f"Read the image from the saved path {img_path}")
    faces = detect_faces(img_path)
    if len(faces) == 0:
        print("No face detected for authentication.")
        return None

    embeddings = extract_face_embeddings(faces)
    if not embeddings:
        print("Failed to extract face embeddings.")
        return None
    
    return embeddings[0]

def authenticate_user_with_cnn(threshold=0.9):
    live_embedding = detect_and_extract_embeddings()
    if live_embedding is None:
        return redirect(url_for('authfail'))

    model, label_encoder = load_keras_model()

    live_embedding = np.expand_dims(live_embedding, axis=0)
    
    prediction = model.predict(live_embedding)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = np.argmax(prediction)
    probability = prediction[0][predicted_label]
    

    user_id = label_encoder.inverse_transform([predicted_class])[0]
    print(f"User ID: {user_id}")
    username = get_username_from_csv(user_id)
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Predicted User: {username}")
    print(f"Prediction Probability: {probability:.2f}")
    
    release_camera()
    
    if probability >= threshold:
        resp = make_response(redirect(url_for('authsuccess')))
        resp.set_cookie('user_id', str(user_id))
        resp.set_cookie('username', username)
        resp.set_cookie('time', time)
        return resp
    else:
        return redirect(url_for('authfail'))