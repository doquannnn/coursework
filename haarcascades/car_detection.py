import cv2
import os

car_weights = [("hogcascade_cars_sideview.xml", "sideview"),
 ("lbpcascade_cars_frontbackview.xml", "frontbackview")]
def detect(image_path, car_weight):
    # Load the cascade
    car_cascade = cv2.CascadeClassifier(
        os.path.join('weights', car_weight[0]))
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect cars
    cars = car_cascade.detectMultiScale(gray, 1.15, 2)
    # Draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imwrite(f'car/{car_weight[1]}.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()


# detectMultiScale params:
# sideview: 1.1, 3
# frontbackview: 1.15, 2
# detect('images/cars.jpeg', car_weights[0])
detect('images/front_car.jpeg', car_weights[1])



