import cv2
import os

body_weights = [("haarcascade_fullbody.xml", "fullbody"),
 ("haarcascade_lowerbody.xml", "lowerbody"),
 ("haarcascade_upperbody.xml", "upperbody")]
def detect(image_path, body_weight):
    # Load the cascade
    body_cascade = cv2.CascadeClassifier(
        os.path.join('weights', body_weight[0]))
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect bodies
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imwrite(f'body/{body_weight[1]}.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()

# detectMultiScale params:
# fullbody, lowerbody, upperbody: 1.1, 4
detect('images/group.jpeg', body_weights[2])
