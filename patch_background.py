# import cv2
# import torch
# import numpy as np
# import os
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from models.common import DetectMultiBackend, AutoShape

# # Config value
# frames_folder = "/home/thieugt95/code/yolov9/data_ext/ped2/training/frames/01_Train"
# gendata_folder = "/home/thieugt95/code/yolov9/data_ext/gendata/01_background"
# extracted_image_path = "/home/thieugt95/code/yolov9/data_ext/extracted_region1.jpg"
# conf_threshold = 0.5
# tracking_class = 0

# # Ensure the output directory exists
# os.makedirs(gendata_folder, exist_ok=True)

# # Initialize DeepSort
# tracker = DeepSort(max_age=30)

# # Initialize YOLOv9
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True)
# model = AutoShape(model)

# # Load classname from classes.names file
# with open("data_ext/classes.names") as f:
#     class_names = f.read().strip().split('\n')

# colors = np.random.randint(0, 255, size=(len(class_names), 3))
# tracks = []

# # Load the extracted region image
# extracted_region = cv2.imread(extracted_image_path)
# if extracted_region is None:
#     raise FileNotFoundError(f"Extracted region image not found at {extracted_image_path}")

# # Get list of frames
# frames_list = sorted([os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.jpg') or frame.endswith('.png')])

# # Track a specific object
# target_track_id = None

# # Process each frame
# for idx, frame_path in enumerate(frames_list):
#     frame = cv2.imread(frame_path)
#     if frame is None:
#         continue

#     # Detect objects in the frame
#     results = model(frame)

#     detect = []
#     for detect_object in results.pred[0]:
#         label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
#         x1, y1, x2, y2 = map(int, bbox)
#         class_id = int(label)

#         if tracking_class is None:
#             if confidence < conf_threshold:
#                 continue
#         else:
#             if class_id != tracking_class or confidence < conf_threshold:
#                 continue

#         detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

#     # Update tracks with DeepSort
#     tracks = tracker.update_tracks(detect, frame=frame)

#     # Draw rectangles and labels on the frame
#     for track in tracks:
#         if track.is_confirmed():
#             track_id = track.track_id
#             ltrb = track.to_ltrb()
#             class_id = track.get_det_class()
#             x1, y1, x2, y2 = map(int, ltrb)

#             # If target_track_id is not set, assign the first track_id
#             if target_track_id is None:
#                 target_track_id = track_id

#             # Process only the target track
#             if track_id == target_track_id:
#                 # Print the bounding box coordinates
#                 print(f"Frame {idx + 1}: Bounding box for track_id {track_id}: ({x1}, {y1}, {x2}, {y2})")

#                 # Resize the extracted region to match the detected bounding box size
#                 region_resized = cv2.resize(extracted_region, (x2 - x1, y2 - y1))

#                 # Paste the resized extracted region into the frame
#                 frame[y1:y2, x1:x2] = region_resized

#     # Save the modified frame to the output folder
#     output_frame_path = os.path.join(gendata_folder, f"{idx + 1:04d}.jpg")
#     cv2.imwrite(output_frame_path, frame)

# cv2.destroyAllWindows()
