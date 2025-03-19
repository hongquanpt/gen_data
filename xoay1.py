import cv2
import torch
import numpy as np
import os
import random
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


# --- Hàm xoay ảnh với tính toán kích thước "bound" (giữ nguyên toàn bộ nội dung) và tạo mặt nạ ---
def rotate_image_bound_with_mask(image, angle):
    """
    Xoay ảnh quanh tâm và trả về ảnh xoay với kích thước mới (bound)
    kèm theo mặt nạ (mask) chỉ vùng có dữ liệu, tránh vùng padding màu đen.
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # Tính kích thước ảnh xoay mới
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # Điều chỉnh ma trận sao cho ảnh xoay được căn giữa
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))
    # Tạo mặt nạ: toàn bộ ảnh gốc đều có giá trị 255, sau đó xoay giống như ảnh đối tượng
    mask = np.ones((h, w), dtype=np.uint8) * 255
    rotated_mask = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
    return rotated, rotated_mask

# --- Hàm xóa nền (không sử dụng vận tốc) ---
def remove_background(nth_person, frames_folder, save_dir, classes_file, weights_path):
    """
    Phát hiện, theo dõi và xóa đối tượng khỏi nền.
    Các bước:
      - Đọc các khung hình từ thư mục frames_folder.
      - Phát hiện đối tượng (ví dụ: người) sử dụng YOLOv9.
      - Theo dõi đối tượng với DeepSort.
      - Xóa đối tượng ra khỏi nền dùng Mask-RCNN + DeepFill.
    Kết quả: lưu các khung hình đã inpaint vào save_dir và trả về danh sách tracking_data.
    """
    conf_threshold = 0.5
    tracking_class = 0  # Ví dụ: theo dõi đối tượng "person"

    os.makedirs(save_dir, exist_ok=True)

    # Khởi tạo DeepSort
    tracker = DeepSort(max_age=30)

    # Khởi tạo YOLOv9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights=weights_path, device=device, fuse=True)
    model = AutoShape(model)

    # Đọc tên lớp từ file
    with open(classes_file) as f:
        class_names = f.read().strip().split('\n')

    # Lấy danh sách các khung hình
    frames_list = sorted([
        os.path.join(frames_folder, frame) 
        for frame in os.listdir(frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])

    # Khởi tạo các mô hình bổ sung: Mask-RCNN và DeepFill
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    rcnn = rcnn.eval()

    deepfill_weights_path = "./models/states_pt_places2.pth"
    deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

    target_track_id = None
    tracking_data = []

    for idx, frame_path in enumerate(frames_list):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Không thể đọc khung hình: {frame_path}")
            continue

        # Phát hiện đối tượng trong khung hình
        results = model(frame)
        detect = []
        for detect_object in results.pred[0]:
            if len(detect_object) < 6:
                continue
            x1, y1, x2, y2, confidence, class_id = detect_object[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)

            if tracking_class is not None:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue
            elif confidence < conf_threshold:
                continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật track với DeepSort
        tracks = tracker.update_tracks(detect, frame=frame)
        person_count = 0

        if target_track_id is None:
            for track in tracks:
                if track.is_confirmed() and track.get_det_class() == tracking_class:
                    person_count += 1
                    if person_count == nth_person:
                        target_track_id = track.track_id
                        print(f"Đã thiết lập ID đối tượng mục tiêu cho người thứ {nth_person}: {target_track_id}")
                        break

        if target_track_id is not None:
            for track in tracks:
                if track.is_confirmed() and track.track_id == target_track_id:
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    tracking_data.append({'frame_idx': idx, 'bbox': (x1, y1, x2, y2)})

                    # Mở rộng hộp bao để có thêm biên xử lý
                    x1_new = max(0, x1 - 5)
                    y1_new = max(0, y1 - 5)
                    x2_new = min(frame.shape[1], x2 + 5)
                    y2_new = min(frame.shape[0], y2 + 5)
                    rectangle = [x1_new, y1_new, x2_new, y2_new]

                    # Xóa đối tượng khỏi khung hình bằng inpaint (mask-RCNN + DeepFill)
                    obj_remove_model = ObjectRemove(
                        segmentModel=rcnn,
                        rcnn_transforms=transforms,
                        inpaintModel=deepfill,
                        image_path=frame_path,
                        box=rectangle
                    )
                    inpainted_output = obj_remove_model.run()

                    if isinstance(inpainted_output, torch.Tensor):
                        inpainted_output = inpainted_output.permute(1, 2, 0).detach().cpu().numpy()
                    if inpainted_output.dtype != np.uint8:
                        inpainted_output = (inpainted_output * 255).clip(0, 255).astype(np.uint8)
                    #inpainted_output = cv2.cvtColor(inpainted_output, cv2.COLOR_RGB2BGR)    #chinh màu
                    inpainted_frame_path = os.path.join(save_dir, os.path.basename(frame_path))
                    cv2.imwrite(inpainted_frame_path, inpainted_output)
                    print(f"Đã lưu khung hình đã xóa đối tượng vào: {inpainted_frame_path}")
                    break

    print("Hoàn thành xóa nền.")
    return tracking_data


# --- Hàm tính gia tốc góc từ bounding box (dùng chiều cao của box làm đại lượng L) ---
def compute_angular_acceleration(bbox, g=9.81):
    """
    Tính gia tốc góc (α) dựa trên chiều cao của đối tượng trong bbox.
    Giả sử đối tượng được mô hình hóa như một thanh đồng nhất quay quanh một đầu.
    Với:
      I = (m * L^2) / 3
      τ = m * g * (L/2)
      => α = τ / I = (3 * g) / (2 * L)
    Nếu L <= 0, trả về 0.
    """
    x1, y1, x2, y2 = bbox
    L = y2 - y1
    if L <= 0:
        return 0
    alpha = (3 * g) / (2 * L)
    return alpha


# --- Hàm dán đối tượng với hiệu ứng xoay (đối tượng bị ngã) sử dụng tính toán góc xoay dựa trên gia tốc góc ---
def paste_object_with_rotation_effect(nth_person, tracking_data, inpainted_frames_folder, original_frames_folder, gendata_folder, max_rotation_angle=90):
    """
    Với mỗi khung hình theo thông tin tracking_data:
      - Trích xuất đối tượng từ khung hình gốc dựa trên bbox.
      - Tính toán góc xoay dựa trên công thức chuyển động quay với FPS = 24 (dt=1/24).
      - Xoay đối tượng bằng hàm rotate_image_bound_with_mask, tạo ra hình xoay có kích thước mới và mặt nạ.
      - Dán đối tượng đã xoay lên khung hình nền (inpainted) sao cho tâm của đối tượng xoay trùng với tâm của bbox ban đầu.
      - Nếu góc xoay đạt đến max_rotation_angle thì dừng xử lý các khung hình tiếp theo.
    """
    os.makedirs(gendata_folder, exist_ok=True)

    inpainted_frames_list = sorted([
        os.path.join(inpainted_frames_folder, frame) 
        for frame in os.listdir(inpainted_frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])
    original_frames_list = sorted([
        os.path.join(original_frames_folder, frame) 
        for frame in os.listdir(original_frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])

    # Khởi tạo thời gian giữa các frame (dt) từ FPS = 24
    dt = 1.0 / 24.0
    current_angle = 10.0             # Góc xoay ban đầu
    current_angular_velocity = 0.0    # Vận tốc góc ban đầu

    for data in tracking_data:
        frame_index = data['frame_idx']
        bbox = data['bbox']

        if frame_index >= len(inpainted_frames_list) or frame_index >= len(original_frames_list):
            print(f"frame_index {frame_index} vượt quá phạm vi khung hình.")
            continue

        inpainted_frame = cv2.imread(inpainted_frames_list[frame_index])
        original_frame = cv2.imread(original_frames_list[frame_index])
        if inpainted_frame is None or original_frame is None:
            print(f"Lỗi khi đọc khung hình tại index {frame_index}.")
            continue

        x1, y1, x2, y2 = bbox
        # Trích xuất đối tượng từ khung hình gốc theo bbox
        extracted_box = original_frame[y1:y2, x1:x2]
        if extracted_box.size == 0:
            print(f"Hộp trích xuất rỗng từ khung hình {frame_index}.")
            continue

        # Tính gia tốc góc dựa trên kích thước bbox (chiều cao của box)
        alpha = compute_angular_acceleration(bbox)
        # Cập nhật vận tốc góc và góc xoay theo công thức:
        # ω_new = ω_old + α * dt
        # θ_new = θ_old + ω_old * dt + 0.5 * α * dt^2
        # Tính delta góc xoay theo công thức vật lý
        delta_angle = current_angular_velocity * dt + 0.5 * alpha * (dt ** 2)

        # Nhân tỉ lệ với độ thay đổi của góc xoay (chỉ nhân delta chứ không nhân vận tốc)
        scale_factor = 50.0  # Ví dụ, nhân 10 lần để hiệu ứng xoay rõ ràng hơn
        scaled_delta_angle = delta_angle * scale_factor

        # Cập nhật góc xoay với delta đã được nhân tỉ lệ
        new_angle = current_angle + scaled_delta_angle

        # Cập nhật vận tốc góc theo công thức ban đầu (không thay đổi)
        new_angular_velocity = current_angular_velocity + alpha * dt

        # Giới hạn góc xoay không vượt quá max_rotation_angle
        angle = min(new_angle, max_rotation_angle)

        # Cập nhật giá trị cho vòng lặp sau
        current_angle = new_angle
        current_angular_velocity = new_angular_velocity


        print(f"Frame {frame_index}: angle = {angle:.2f}°, angular velocity = {current_angular_velocity:.4f}°/frame, alpha = {alpha:.4f}°/frame²")

        # Xoay đối tượng với kích thước bound và nhận thêm mặt nạ (để không dán vùng padding)
        rotated_box, rotated_mask = rotate_image_bound_with_mask(extracted_box, angle)
        (rh, rw) = rotated_box.shape[:2]

        # Tính tâm của bbox ban đầu (để dán đối tượng xoay sao cho tâm khớp)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Tính vị trí dán: top-left sao cho tâm của rotated_box trùng với center_x, center_y
        paste_x1 = center_x - rw // 2
        paste_y1 = center_y - rh // 2
        paste_x2 = paste_x1 + rw
        paste_y2 = paste_y1 + rh

        # Điều chỉnh nếu vùng dán vượt ngoài kích thước của inpainted_frame
        frame_h, frame_w = inpainted_frame.shape[:2]
        dest_x1 = max(paste_x1, 0)
        dest_y1 = max(paste_y1, 0)
        dest_x2 = min(paste_x2, frame_w)
        dest_y2 = min(paste_y2, frame_h)

        # Tính vùng nguồn trên rotated_box tương ứng với vùng dán
        src_x1 = dest_x1 - paste_x1  # Nếu paste_x1 < 0 thì src_x1 = -paste_x1
        src_y1 = dest_y1 - paste_y1
        src_x2 = src_x1 + (dest_x2 - dest_x1)
        src_y2 = src_y1 + (dest_y2 - dest_y1)

        if dest_x2 > dest_x1 and dest_y2 > dest_y1:
            src_rotated_box = rotated_box[src_y1:src_y2, src_x1:src_x2]
            src_rotated_mask = rotated_mask[src_y1:src_y2, src_x1:src_x2]
            # Mở rộng mặt nạ thành 3 kênh để thực hiện phép gán theo từng kênh màu
            mask_expanded = src_rotated_mask[:, :, np.newaxis]
            dest_roi = inpainted_frame[dest_y1:dest_y2, dest_x1:dest_x2]
            combined = np.where(mask_expanded > 0, src_rotated_box, dest_roi)
            inpainted_frame[dest_y1:dest_y2, dest_x1:dest_x2] = combined
        else:
            print(f"Vùng dán không hợp lệ tại frame {frame_index}.")
            continue

        output_frame_path = os.path.join(gendata_folder, os.path.basename(original_frames_list[frame_index]))
        cv2.imwrite(output_frame_path, inpainted_frame)
        print(f"Đã lưu khung hình có đối tượng xoay tại: {output_frame_path}")

        # Nếu góc xoay đã đạt đến max_rotation_angle, dừng xử lý các khung hình tiếp theo.
        if angle >= max_rotation_angle:
            print("Đã đạt đến 90 độ xoay. Dừng xử lý các khung hình tiếp theo.")
            break

    print("Hoàn thành xử lý dán đối tượng với hiệu ứng xoay (ngã).")


# --- Luồng chính ---
def main():
    parser = argparse.ArgumentParser(
        description="Xử lý xóa nền và dán đối tượng với hiệu ứng xoay (đối tượng bị ngã) mà không sử dụng kích thước box cũ."
    )
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Đường dẫn tới thư mục chứa các khung hình gốc (ví dụ: ./data_ext/training/01).')
    args = parser.parse_args()

    input_folder = args.input_folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    background_save_dir = os.path.join("./data_ext/gendata/background", input_folder_name)
    rotation_effect_save_dir = os.path.join("./data_ext/gendata/rotation_effect", input_folder_name)

    classes_file = "data_ext/classes.names"
    weights_path = "./weights/yolov9-e.pt"
    nth_person = 1  # Chọn người thứ 1

    # Gọi hàm xóa nền (không sử dụng tham số vận tốc)
    tracking_data = remove_background(
        nth_person,
        frames_folder=input_folder,
        save_dir=background_save_dir,
        classes_file=classes_file,
        weights_path=weights_path
    )

    # Gọi hàm dán đối tượng với hiệu ứng xoay, sử dụng cập nhật góc xoay dựa trên gia tốc góc thực tế
    paste_object_with_rotation_effect(
        nth_person,
        tracking_data,
        inpainted_frames_folder=background_save_dir,
        original_frames_folder=input_folder,
        gendata_folder=rotation_effect_save_dir,
        max_rotation_angle=90      # Góc xoay tối đa là 90 độ
    )

if __name__ == "__main__":
    main()
