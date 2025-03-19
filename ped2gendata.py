import cv2
import torch
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config values
frames_folder = "data_ext/01_Train"  # Folder containing the frames
output_folder = "/home/thieugt95/Desktop/ped2"  # Folder to save the modified images
modified_image_path = "/home/thieugt95/code/yolov9/data_ext/modified_image.jpg"

conf_threshold = 0.5
tracking_class = 0  # Assuming class '0' corresponds to 'person' in your classes.names file

# Initialize DeepSort
tracker = DeepSort(max_age=30)

# Initialize YOLOv9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True)
model = AutoShape(model)

# Load classname from classes.names file
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Get list of frames
frames_list = sorted([os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.jpg') or frame.endswith('.png')])

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Track a specific object
target_track_id = None

# Indices of frames to extract
frame_indices_to_extract = [3, 7, 11, 15]

# Process each frame
for idx, frame_path in enumerate(frames_list):
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    # Detect objects in the frame
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update tracks with DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # Draw rectangles and labels on the frame
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)

            # If target_track_id is not set, assign the first track_id
            if target_track_id is None:
                target_track_id = track_id

            # Process only the target track
            if track_id == target_track_id:
                color = colors[class_id]
                B, G, R = map(int, color)
                label = "{}-{}".format(class_names[class_id], track_id)
        
                
                # Print the bounding box coordinates
                print(f"Frame {idx + 1}: Bounding box for track_id {track_id}: ({x1}, {y1}, {x2}, {y2})")

                # If the frame is one of the frames to extract, process it
                if idx in frame_indices_to_extract:
                    # Load the modified image
                    modified_image = cv2.imread(modified_image_path)
                    if modified_image is None:
                        raise FileNotFoundError(f"Modified image not found at {modified_image_path}")

                    # Extract the region of the tracked object
                    extracted_region = frame[y1:y2, x1:x2]

                    # Ensure the extracted region size fits into the modified image at the same location
                    region_height, region_width = extracted_region.shape[:2]
                    if (y2 - y1, x2 - x1) != (region_height, region_width):
                        raise ValueError("The extracted region size does not match the target region size.")

                    # Paste the extracted region into the modified image at the same coordinates
                    modified_image[y1:y2, x1:x2] = extracted_region

                    # Save the modified image to the output folder
                    output_image_path = os.path.join(output_folder, f"modified_frame_{idx + 1}.jpg")
                    cv2.imwrite(output_image_path, modified_image)
                    print(f"Saved modified frame {idx + 1} to {output_image_path}")

    # Display the frame (optional)
    cv2.imshow("Object Tracking", frame)
    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
