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
    kèm theo mask với nội suy mềm (anti-aliased) để giảm hiện tượng viền đen.
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # Tính kích thước ảnh xoay mới
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # Điều chỉnh ma trận để căn giữa ảnh xoay
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Xoay ảnh với borderMode replicates các giá trị biên, tránh đen xung quanh
    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tạo mask ban đầu với giá trị 1, sau đó xoay theo cùng ma trận M
    mask = np.ones((h, w), dtype=np.float32)
    rotated_mask = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Chuyển mask về dạng 0-255 (vẫn giữ nội suy mềm)
    rotated_mask = (rotated_mask * 255).astype(np.uint8)

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
                    inpainted_output = cv2.cvtColor(inpainted_output, cv2.COLOR_RGB2BGR)
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

# --- Hàm dán đối tượng với hiệu ứng xoay (đối tượng bị ngã) ---
def paste_object_with_rotation_effect(nth_person, tracking_data, inpainted_frames_folder, original_frames_folder, gendata_folder, max_rotation_angle=90):
    """
    Với mỗi khung hình theo thông tin tracking_data:
      - Trích xuất đối tượng từ khung hình gốc dựa trên bbox.
      - Tính toán góc xoay dựa trên công thức chuyển động quay với FPS = 24 (dt=1/24).
      - Xoay đối tượng bằng hàm rotate_image_bound_with_mask, tạo ra hình xoay có kích thước mới và mask mềm.
      - Dán đối tượng đã xoay lên khung hình nền (inpainted) bằng alpha blending để giảm hiện tượng viền đen.
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

    # Thời gian giữa các frame (dt) với FPS = 24
    dt = 1.0 / 24.0
    current_angle = 10.0             # Góc xoay ban đầu
    current_angular_velocity = 0.27    # Vận tốc góc ban đầu

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
        # Cập nhật vận tốc góc và góc xoay theo công thức vật lý:
        delta_angle = current_angular_velocity * dt + 0.5 * alpha * (dt ** 2)
        scale_factor = 30.0  # Nhân tỉ lệ để hiệu ứng xoay rõ ràng hơn
        scaled_delta_angle = delta_angle * scale_factor

        new_angle = current_angle + scaled_delta_angle
        new_angular_velocity = current_angular_velocity + alpha * dt
        angle = min(new_angle, max_rotation_angle)

        # Cập nhật giá trị cho vòng lặp sau
        current_angle = new_angle
        current_angular_velocity = new_angular_velocity

        print(f"Frame {frame_index}: angle = {angle:.2f}°, angular velocity = {current_angular_velocity:.4f}°/frame, alpha = {alpha:.4f}°/frame²")

        # Xoay đối tượng với mask mềm
        rotated_box, rotated_mask = rotate_image_bound_with_mask(extracted_box, angle)
        (rh, rw) = rotated_box.shape[:2]

        # Tính tâm của bbox ban đầu
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Tính vị trí dán sao cho tâm của rotated_box trùng với center của bbox
        paste_x1 = center_x - rw // 2
        paste_y1 = center_y - rh // 2
        paste_x2 = paste_x1 + rw
        paste_y2 = paste_y1 + rh

        frame_h, frame_w = inpainted_frame.shape[:2]
        dest_x1 = max(paste_x1, 0)
        dest_y1 = max(paste_y1, 0)
        dest_x2 = min(paste_x2, frame_w)
        dest_y2 = min(paste_y2, frame_h)

        # Tính vùng nguồn tương ứng từ rotated_box và rotated_mask
        src_x1 = dest_x1 - paste_x1
        src_y1 = dest_y1 - paste_y1
        src_x2 = src_x1 + (dest_x2 - dest_x1)
        src_y2 = src_y1 + (dest_y2 - dest_y1)

        if dest_x2 > dest_x1 and dest_y2 > dest_y1:
            src_rotated_box = rotated_box[src_y1:src_y2, src_x1:src_x2]
            src_rotated_mask = rotated_mask[src_y1:src_y2, src_x1:src_x2]

            # (Tùy chọn) Dùng phép xói mòn mask để loại bỏ viền nhẹ
            kernel = np.ones((3, 3), np.uint8)
            src_rotated_mask = cv2.erode(src_rotated_mask, kernel, iterations=1)

            # Chuyển mask sang dạng float [0,1] để thực hiện alpha blending
            alpha_mask = src_rotated_mask.astype(np.float32) / 255.0

            dest_roi = inpainted_frame[dest_y1:dest_y2, dest_x1:dest_x2].astype(np.float32)
            src_roi = src_rotated_box.astype(np.float32)

            # Thực hiện alpha blending
            blended = (alpha_mask[..., np.newaxis] * src_roi +
                       (1 - alpha_mask[..., np.newaxis]) * dest_roi)
            blended = blended.clip(0, 255).astype(np.uint8)
            inpainted_frame[dest_y1:dest_y2, dest_x1:dest_x2] = blended
        else:
            print(f"Vùng dán không hợp lệ tại frame {frame_index}.")
            continue

        output_frame_path = os.path.join(gendata_folder, os.path.basename(original_frames_list[frame_index]))
        cv2.imwrite(output_frame_path, inpainted_frame)
        print(f"Đã lưu khung hình có đối tượng xoay tại: {output_frame_path}")

        # Nếu góc xoay đã đạt đến max_rotation_angle, dừng xử lý
        if angle >= max_rotation_angle:
            print("Đã đạt đến 90 độ xoay. Dừng xử lý các khung hình tiếp theo.")
            break

    print("Hoàn thành xử lý dán đối tượng với hiệu ứng xoay (ngã).")

def paste_object_with_sudden_velocity_change(nth_person, tracking_data, v1_real, v2_real, inpainted_frames_folder, original_frames_folder, gendata_folder):
    # ------------------------- Cấu Hình -------------------------
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(gendata_folder, exist_ok=True)

    # ------------------------- Khởi Tạo Biến -------------------------
    FPS = 24  # Số khung hình trên giây của video
    frame_idx = 0  # Chỉ số khung hình trong tracking_data

    # Lấy danh sách các khung hình
    inpainted_frames_list = sorted([
        os.path.join(inpainted_frames_folder, frame) for frame in os.listdir(inpainted_frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])

    original_frames_list = sorted([
        os.path.join(original_frames_folder, frame) for frame in os.listdir(original_frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])

    while frame_idx < len(tracking_data):
        data = tracking_data[frame_idx]
        frame_index = data['frame_idx']
        bbox = data['bbox']

        # Kiểm tra nếu frame_index hợp lệ
        if frame_index >= len(inpainted_frames_list):
            print(f"frame_index {frame_index} vượt quá phạm vi danh sách inpainted_frames_list ({len(inpainted_frames_list)}). Bỏ qua khung hình này.")
            frame_idx += 1
            continue

        # Đọc khung hình đã xóa đối tượng
        inpainted_frame_path = inpainted_frames_list[frame_index]
        inpainted_frame = cv2.imread(inpainted_frame_path)
        if inpainted_frame is None:
            print(f"Không thể đọc khung hình đã xóa đối tượng: {inpainted_frame_path}")
            frame_idx += 1
            continue

        # Đọc khung hình gốc
        if frame_index >= len(original_frames_list):
            print(f"frame_index {frame_index} vượt quá phạm vi danh sách original_frames_list ({len(original_frames_list)}). Bỏ qua khung hình này.")
            frame_idx += 1
            continue

        original_frame_path = original_frames_list[frame_index]
        original_frame = cv2.imread(original_frame_path)
        if original_frame is None:
            print(f"Không thể đọc khung hình gốc: {original_frame_path}")
            frame_idx += 1
            continue

        # Trích xuất hộp bao từ khung hình gốc
        x1, y1, x2, y2 = bbox
        extracted_box = original_frame[y1:y2, x1:x2]
        if extracted_box.size == 0:
            print(f"Hộp trích xuất rỗng từ khung hình {frame_index}.")
            frame_idx += 1
            continue

        # Dán hộp bao lên khung hình đã xóa đối tượng
        y2_paste = min(y2, inpainted_frame.shape[0])
        x2_paste = min(x2, inpainted_frame.shape[1])
        y1_paste = max(y1, 0)
        x1_paste = max(x1, 0)
        paste_height = y2_paste - y1_paste
        paste_width = x2_paste - x1_paste

        # Resize hộp bao để phù hợp với khu vực dán
        resized_box = cv2.resize(extracted_box, (paste_width, paste_height))

        # Dán hộp bao lên khung hình đã xóa đối tượng
        inpainted_frame[y1_paste:y2_paste, x1_paste:x2_paste] = resized_box

        # Lưu khung hình đã chỉnh sửa vào thư mục đầu ra
        output_frame_path = os.path.join(gendata_folder, os.path.basename(original_frame_path))
        cv2.imwrite(output_frame_path, inpainted_frame)
        print(f"Đã lưu khung hình đã chỉnh sửa vào: {output_frame_path}")

        # ------------------------- Tính Số Khung Hình Cần Bỏ Qua -------------------------
        if v1_real > 0:
            skip_ratio = v2_real / v1_real  # Chia theo vận tốc ban đầu v1_real
            frames_to_skip = int(skip_ratio) - 1
            if frames_to_skip < 0:
                frames_to_skip = 0  # Tránh giá trị âm
        else:
            frames_to_skip = 0  # Tránh chia cho 0

        # Cập nhật frame_idx để bỏ qua các khung hình đã xử lý
        frame_idx += frames_to_skip + 1  # +1 để chuyển đến khung hình tiếp theo

       
    print("Hoàn thành xử lý dán đối tượng với vận tốc thay đổi đột ngột.")

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
    sudden_velocity_save_dir = os.path.join("./data_ext/gendata/sudden_velocity", input_folder_name)
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
    print("\n=== BẮT ĐẦU: HIỆU ỨNG TĂNG TỐC ĐỘT NGỘT ===\n")
    v1_real = 1.0  # Vận tốc ban đầu (đơn vị tùy chọn, ví dụ: px/frame)
    v2_real = 4.0  # Vận tốc tăng đột ngột (gấp 4 lần)
    paste_object_with_sudden_velocity_change(
        nth_person,
        tracking_data,
        v1_real=v1_real,
        v2_real=v2_real,
        inpainted_frames_folder=background_save_dir,
        original_frames_folder=input_folder,
        gendata_folder=sudden_velocity_save_dir
    )
if __name__ == "__main__":
    main()
