import cv2
import os


# Define the path to your dataset
dataset_path = 'input'



# Function to extract frames from a video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

# Process all video files in the dataset
for video_file in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video_file)
    output_folder = os.path.join('frames', video_file.split('.')[0])  # Create a folder for frames
    os.makedirs(output_folder, exist_ok=True)
    extract_frames(video_path, output_folder)
