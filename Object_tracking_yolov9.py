import cv2
import torch
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

def track_nth_person(nth_person):
    # Config value
    frames_folder = "./data_ext/02_Train"  # Folder containing the frames

    conf_threshold = 0.5
    tracking_class = 0  # Assuming class '0' corresponds to 'person' in your classes.names file

    # Initialize DeepSort
    tracker = DeepSort(max_age=30)

    # Initialize YOLOv9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights="./weights/yolov9-e.pt", device=device, fuse=True)
    model = AutoShape(model)

    # Load classname from classes.names file
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # Get list of frames
    frames_list = sorted([os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.jpg') or frame.endswith('.png')])

    # Track a specific object
    target_track_id = None

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

        # Counter for number of people detected
        person_count = 0

        # Draw rectangles and labels on the frame
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)

                # Gán target_track_id cho đối tượng người thứ nth_person
                if target_track_id is None:
                    person_count += 1
                    if person_count == nth_person:  # Chọn người thứ nth_person
                        target_track_id = track_id
                        break  # Dừng vòng lặp sau khi chọn đối tượng thứ nth_person

                # Process only the target track
                if track_id == target_track_id:
                    color = colors[class_id]
                    B, G, R = map(int, color)
                    label = "{}-{}".format(class_names[class_id], track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # Print the bounding box coordinates
                    print(f"Frame {idx + 1}: Bounding box for track_id {track_id}: ({x1}, {y1}, {x2}, {y2})")

        # Display the frame
        cv2.imshow("Object Tracking", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

# Chạy hàm với tham số chỉ định người thứ mấy sẽ được theo dõi (ví dụ: người thứ 3)
track_nth_person(4)





