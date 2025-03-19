import cv2
import torch
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config value
frames_folder = "/home/thieugt95/code/yolov9/data_ext/16_Train"
background_folder = "/home/thieugt95/Desktop/gendata/16_background"
gendata_folder = "/home/thieugt95/Desktop/gendata/16"
conf_threshold = 0.5
tracking_class = 0


# Ensure the output directory exists
os.makedirs(gendata_folder, exist_ok=True)


# Initialize DeepSort
tracker = DeepSort(max_age=30)

# Initialize YOLOv9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights="weights/yolov9-e.pt", device=device, fuse=True)
model = AutoShape(model)

# Load classname from classes.names file
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Get list of frames
frames_list = sorted([os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.jpg') or frame.endswith('.png')])
background_frames_list = sorted([os.path.join(background_folder, frame) for frame in os.listdir(background_folder) if frame.endswith('.jpg') or frame.endswith('.png')])

# Track a specific object
target_track_id = None

# Define the intervals for frame extraction (every 4th frame)
intervals = list(range(3, len(frames_list), 4))
extracted_boxes = []

def add_gaussian_noise(image, mean=0, var=0.01):
    """
    Add Gaussian noise to an image.
    
    Parameters:
    image (numpy.ndarray): The input image.
    mean (float): The mean of the Gaussian noise.
    var (float): The variance of the Gaussian noise.
    
    Returns:
    numpy.ndarray: The image with added Gaussian noise.
    """
    sigma = 0
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

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

    # Extract boxes for the specific intervals
    if (idx + 1) in intervals:
        for track in tracks:
            if track.is_confirmed() and track.track_id == target_track_id:
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                extracted_box = frame[y1:y2, x1:x2]
                # Add Gaussian noise to the extracted box
                noisy_box = add_gaussian_noise(extracted_box)
                extracted_boxes.append((noisy_box, (x1, y1, x2, y2)))
                break

    # Set the target track ID for the first time
    for track in tracks:
        if track.is_confirmed() and target_track_id is None:
            target_track_id = track.track_id
            break

# Paste the extracted regions into background frames
for idx, (noisy_box, (x1, y1, x2, y2)) in enumerate(extracted_boxes):
    if idx >= len(background_frames_list):
        break

    background_frame_path = background_frames_list[idx]
    background_frame = cv2.imread(background_frame_path)
    if background_frame is None:
        continue

    # Resize the noisy box to fit the original dimensions
    resized_box = cv2.resize(noisy_box, (x2 - x1, y2 - y1))

    
    # Paste the resized box into the background frame
    background_frame[y1:y2, x1:x2] = resized_box
    
    # Muốn dịch chuyển box đối tượng xuống dưới thì + cả y1 và y2 1 đoạn
    #background_frame[y1 + 40:min(y2 + 40, background_frame.shape[0]), x1:x2] = resized_box

    # Save the modified frame
    output_frame_path = os.path.join(gendata_folder, os.path.basename(background_frame_path))
    cv2.imwrite(output_frame_path, background_frame)
    print(f"Saved modified frame to {output_frame_path}")

cv2.destroyAllWindows()