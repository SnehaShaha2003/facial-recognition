import cv2
import numpy as np
from skimage.feature import hog
import pickle

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Initialize object tracker
tracker = cv2.TrackerCSRT_create()

# Initialize variables for tracking
persons = {}  # Dictionary to store tracking information

# Initialize a dictionary to store the features for each person
person_features = {}

# Open a video file or capture from a camera
cap = cv2.VideoCapture("input\input.mp4")  # Replace with your video file or camera index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clear the image by filling it with the original frame
    display_frame = frame.copy()

    # Perform person detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 0 corresponds to persons
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            person_id = None
            for pid, (px, py, pw, ph, tracker) in persons.items():
                if x >= px and x <= px + pw and y >= py and y <= py + ph:
                    person_id = pid
                    break

            if person_id is None:
                tracker.init(frame, (x, y, w, h))
                person_id = len(persons) + 1

            persons[person_id] = (x, y, w, h, tracker)

            # Extract HOG features for the tracked person
            person_image = frame[y:y + h, x:x + w]

            # Convert the image to grayscale (HOG works on grayscale images)
            gray_person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

            # Extract HOG features from the grayscale image
            hog_features = hog(gray_person_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys')

            # Save the HOG features in the dictionary
            person_features[person_id] = hog_features

    # Update tracking
    for pid, (x, y, w, h, tracker) in list(persons.items()):
        success, new_box = tracker.update(frame)
        if not success:
            del persons[pid]

    # Draw bounding boxes and person IDs on the display_frame
    for pid, (x, y, w, h, _) in persons.items():
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, f'Person {pid}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Person Detection and Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Save the extracted features to a file (e.g., features.pickle)
with open("features.pickle", "wb") as file:
    pickle.dump(person_features, file)
