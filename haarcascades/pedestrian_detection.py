import cv2
import os

def detect(image_path, pedes_weight):
    # Load the cascade
    pedes_cascade = cv2.CascadeClassifier(
        os.path.join("weights", pedes_weight))
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    pedestrians = pedes_cascade.detectMultiScale(gray, 1.1, 5)
    # Draw rectangle around the faces
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imwrite(f'pedestrians/{image_path.split(".")[0]}_result.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()


detect('images/crossing.jpeg', 'pedestrian.xml')
