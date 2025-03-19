import cv2
import torch
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def remove_nth_person(nth_person):
    # Config value
    frames_folder = "./data_ext/01_Train"
    save_dir = "./data_ext/gendata/01_4_background"
    conf_threshold = 0.5
    tracking_class = 0  # Assuming class '0' corresponds to 'person' in your classes.names file

    # Initialize DeepSort
    tracker = DeepSort(max_age=30)

    # Initialize YOLOv9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights="./weights/yolov9-e.pt", device=device, fuse=True)
    model = AutoShape(model)

    # Load class names from classes.names file
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # Get list of frames
    frames_list = sorted([os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.jpg') or frame.endswith('.png')])

    # Track a specific object
    target_track_id = None

    # Initialize Mask-RCNN and DeepFill models
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    rcnn = rcnn.eval()

    # DeepFill model
    deepfill_weights_path = "./models/states_pt_places2.pth"
    deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

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

                # Assign target_track_id to the nth person
                if target_track_id is None:
                    person_count += 1
                    if person_count == nth_person:  # Select the nth person
                        target_track_id = track_id
                        break  # Stop the loop after finding the nth person

                # Process only the target track
                if track_id == target_track_id:
                    color = colors[class_id]
                    B, G, R = map(int, color)
                    label = "{}-{}".format(class_names[class_id], track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Increase bounding box dimensions
                    x1_new = max(0, x1 - 5)
                    y1_new = max(0, y1 - 5)
                    x2_new = min(frame.shape[1], x2 + 5)
                    y2_new = min(frame.shape[0], y2 + 5)

                    # Process for object removal and inpainting with expanded box
                    rectangle = [x1_new, y1_new, x2_new, y2_new]

                    # Initialize ObjectRemove for each frame and bounding box
                    obj_remove_model = ObjectRemove(
                        segmentModel=rcnn,
                        rcnn_transforms=transforms,
                        inpaintModel=deepfill,
                        image_path=frame_path,
                        box=rectangle
                    )

                    # Run the object removal process
                    inpainted_output = obj_remove_model.run()

                    # Convert inpainted_output to a NumPy array if necessary
                    if isinstance(inpainted_output, torch.Tensor):
                        inpainted_output = inpainted_output.permute(1, 2, 0).detach().cpu().numpy()

                    # Ensure inpainted_output is in uint8 format and RGB color
                    if inpainted_output.dtype != np.uint8:
                        inpainted_output = (inpainted_output * 255).clip(0, 255).astype(np.uint8)

                    # Save the inpainted image
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{idx + 1:03}_{track_id}.png")
                    cv2.imwrite(save_path, inpainted_output)

        # Display the frame
        cv2.imshow("Object Tracking", frame)
        # Press 'q' to exit
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

# Run the function with specified person index to remove (e.g., person number 3)
remove_nth_person(4)
