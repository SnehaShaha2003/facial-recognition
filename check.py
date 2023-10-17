import cv2
import numpy as np
from skimage.feature import hog  # Add this import statement
import pickle

# Load the saved HOG features from the file
with open("features.pickle", "rb") as file:
    loaded_features = pickle.load(file)

# Set a similarity threshold for recognition
threshold = 0.1  # Adjust this value as needed

# Define the capture from a camera or video source
cap = cv2.VideoCapture("input\input.mp4")  # Replace with your video file or camera index

# Modify the following section to match the input code
while True:
    # Capture a frame from a camera or video source
    ret, frame = cap.read()
    if not ret:
        break

    # Extract HOG features for the current frame (similar to the input code)
    gray_new_person_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    new_person_hog_features = hog(gray_new_person_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)

    # Compare new person's features with the features of known persons
    for person_id, features in loaded_features.items():
        if len(features) != len(new_person_hog_features):
            continue  # Skip persons with different feature dimensions

        # Calculate similarity (e.g., using cosine similarity)
        similarity = np.dot(new_person_hog_features, features) / (np.linalg.norm(new_person_hog_features) * np.linalg.norm(features))

        if similarity > threshold:
            print(f"Recognized as Person {person_id} with similarity: {similarity:.2f}")

    # Display the frame with recognized persons
    cv2.imshow("Recognized Persons", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
