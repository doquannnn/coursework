import cv2
import os

facial_weights = [("haarcascade_eye.xml", "eye"),
 ("haarcascade_mcs_mouth.xml", "mouth"),
 ("haarcascade_mcs_nose.xml", "nose")]
def detect(image_path, facial_weight):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        os.path.join('weights', facial_weight[0]))
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.5, 8)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imwrite(f'facial_features/{facial_weight[1]}.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()

# detectMultiScale params:
# eye: 1.1, 4
# mouth, nose: 1.5, 8
detect('images/my_face.jpeg', facial_weights[2])
