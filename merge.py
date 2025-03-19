import cv2
import torch
import numpy as np
import os
import random
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import math

def remove_background(nth_person, v2_real, a_real):
    # ------------------------- Cấu Hình -------------------------
    frames_folder = "./data_ext/01_Train"
    save_dir = "./data_ext/gendata/01_background"  # Thư mục lưu khung hình đã xóa đối tượng
    conf_threshold = 0.5
    tracking_class = 0  

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------- Khởi Tạo Mô Hình -------------------------
    # Khởi tạo DeepSort
    tracker = DeepSort(max_age=30)

    # Khởi tạo YOLOv9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights="./weights/yolov9-e.pt", device=device, fuse=True)
    model = AutoShape(model)

    # Tải tên các lớp từ tệp classes.names
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # Lấy danh sách các khung hình
    frames_list = sorted([
        os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])

    # ------------------------- Khởi Tạo Các Mô Hình Bổ Sung -------------------------
    # Khởi tạo Mask-RCNN và DeepFill cho việc loại bỏ đối tượng
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    rcnn = rcnn.eval()

    # Mô hình DeepFill
    deepfill_weights_path = "./models/states_pt_places2.pth"
    deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

    # ------------------------- Khởi Tạo Biến -------------------------
    target_track_id = None
    tracking_data = []  # Danh sách lưu thông tin hộp bao của đối tượng mục tiêu

    # ------------------------- Hiệu Chuẩn Tỷ Lệ -------------------------
    FPS = 24  # Số khung hình trên giây của video
    v1_real = 0.83  # m/s (vận tốc đi bộ)

    print(f"Vận tốc đi bộ: {v1_real:.2f} m/s")
    print(f"Vận tốc tối đa (được chọn ngẫu nhiên): {v2_real:.2f} m/s")
    print(f"Gia tốc: {a_real:.2f} m/s²")

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
                continue  # Bỏ qua nếu không đủ thông tin
            x1, y1, x2, y2, confidence, class_id = detect_object[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)

            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật các tracks với DeepSort
        tracks = tracker.update_tracks(detect, frame=frame)

        # Khởi tạo biến đếm số người
        person_count = 0

        # Xác định target_track_id cho người thứ nth_person
        if target_track_id is None:
            for track in tracks:
                if track.is_confirmed() and track.get_det_class() == tracking_class:
                    person_count += 1
                    if person_count == nth_person:
                        target_track_id = track.track_id
                        print(f"Đã thiết lập ID đối tượng mục tiêu cho người thứ {nth_person}: {target_track_id}")
                        break

        # Nếu đang theo dõi đối tượng mục tiêu
        if target_track_id is not None:
            for track in tracks:
                if track.is_confirmed() and track.track_id == target_track_id:
                    # Lưu thông tin hộp bao
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    tracking_data.append({'frame_idx': idx, 'bbox': (x1, y1, x2, y2)})

                    # Mở rộng kích thước hộp bao một chút
                    x1_new = max(0, x1 - 5)
                    y1_new = max(0, y1 - 5)
                    x2_new = min(frame.shape[1], x2 + 5)
                    y2_new = min(frame.shape[0], y2 + 5)
                    rectangle = [x1_new, y1_new, x2_new, y2_new]

                    # Thực hiện loại bỏ đối tượng và tô màu lại
                    obj_remove_model = ObjectRemove(
                        segmentModel=rcnn,
                        rcnn_transforms=transforms,
                        inpaintModel=deepfill,
                        image_path=frame_path,
                        box=rectangle
                    )
                    inpainted_output = obj_remove_model.run()

                    # Chuyển đổi inpainted_output sang NumPy array nếu cần
                    if isinstance(inpainted_output, torch.Tensor):
                        inpainted_output = inpainted_output.permute(1, 2, 0).detach().cpu().numpy()

                    # Đảm bảo inpainted_output ở định dạng uint8 và màu RGB
                    if inpainted_output.dtype != np.uint8:
                        inpainted_output = (inpainted_output * 255).clip(0, 255).astype(np.uint8)

                    # Lưu khung hình đã xóa đối tượng
                    inpainted_frame_path = os.path.join(save_dir, os.path.basename(frame_path))
                    cv2.imwrite(inpainted_frame_path, inpainted_output)
                    print(f"Đã lưu khung hình đã xóa đối tượng vào: {inpainted_frame_path}")

                    break  # Chỉ xử lý đối tượng mục tiêu

    print("Hoàn thành xóa nền.")
    return tracking_data, v1_real  # Trả về tracking_data và v1_real

def paste_object_with_acceleration(nth_person, tracking_data, v1_real, v2_real, a_real):
    # ------------------------- Cấu Hình -------------------------
    inpainted_frames_folder = "./data_ext/gendata/01_background"  # Thư mục chứa khung hình đã xóa đối tượng
    original_frames_folder = "./data_ext/01_Train"  # Thư mục chứa khung hình gốc
    gendata_folder = "./data_ext/gendata/01_acceleration"  # Thư mục lưu khung hình đầu ra cuối cùng

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(gendata_folder, exist_ok=True)

    # ------------------------- Hiệu Chuẩn Tỷ Lệ -------------------------
    FPS = 24  # Số khung hình trên giây của video

    print(f"Vận tốc đi bộ: {v1_real:.2f} m/s")
    print(f"Vận tốc tối đa (được chọn ngẫu nhiên): {v2_real:.2f} m/s")
    print(f"Gia tốc: {a_real:.2f} m/s²")

    # ------------------------- Hàm Thêm Nhiễu Gaussian -------------------------
    def add_gaussian_noise(image, mean=0, var=0.01):
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
        noisy = cv2.add(image, gauss)
        return noisy

    # ------------------------- Khởi Tạo Biến -------------------------
    acceleration_phase = 'accelerating'  # 'accelerating', 'constant', 'decelerating'
    constant_velocity_frame_count = 0  # Đếm số khung hình ở vận tốc v2_real
    frame_idx = 0  # Chỉ số khung hình trong tracking_data
    current_velocity = v1_real  # Vận tốc ban đầu

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

        # Đọc khung hình đã xóa đối tượng
        inpainted_frame_path = inpainted_frames_list[frame_index]
        inpainted_frame = cv2.imread(inpainted_frame_path)
        if inpainted_frame is None:
            print(f"Không thể đọc khung hình đã xóa đối tượng: {inpainted_frame_path}")
            frame_idx += 1
            continue

        # Đọc khung hình gốc
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

        # Thêm nhiễu Gaussian vào hộp trích xuất
        noisy_box = add_gaussian_noise(extracted_box)

        # Dán hộp nhiễu lên khung hình đã xóa đối tượng
        y2_paste = min(y2, inpainted_frame.shape[0])
        x2_paste = min(x2, inpainted_frame.shape[1])
        y1_paste = max(y1, 0)
        x1_paste = max(x1, 0)
        paste_height = y2_paste - y1_paste
        paste_width = x2_paste - x1_paste

        # Resize hộp nhiễu để phù hợp với khu vực dán
        resized_box = cv2.resize(noisy_box, (paste_width, paste_height))

        # Dán hộp nhiễu lên khung hình đã xóa đối tượng
        inpainted_frame[y1_paste:y2_paste, x1_paste:x2_paste] = resized_box

        # Lưu khung hình đã chỉnh sửa vào thư mục đầu ra
        output_frame_path = os.path.join(gendata_folder, os.path.basename(original_frame_path))
        cv2.imwrite(output_frame_path, inpainted_frame)
        print(f"Đã lưu khung hình đã chỉnh sửa vào: {output_frame_path}")

        # ------------------------- Tính Vận Tốc Hiện Tại -------------------------
        if acceleration_phase == 'accelerating':
            current_velocity += a_real / FPS
            if current_velocity >= v2_real:
                current_velocity = v2_real
                acceleration_phase = 'constant'
                constant_velocity_frame_count = 0
            print(f"Frame {frame_idx}: Vận tốc hiện tại (tăng tốc) = {current_velocity:.2f} m/s")
        elif acceleration_phase == 'constant':
            current_velocity = v2_real
            constant_velocity_frame_count += 1
            print(f"Frame {frame_idx}: Vận tốc hiện tại (ổn định) = {current_velocity:.2f} m/s")
            if constant_velocity_frame_count >= 20:
                acceleration_phase = 'decelerating'
        elif acceleration_phase == 'decelerating':
            current_velocity -= a_real / FPS
            if current_velocity <= v1_real:
                current_velocity = v1_real
                acceleration_phase = 'constant_at_v1'
            print(f"Frame {frame_idx}: Vận tốc hiện tại (giảm tốc) = {current_velocity:.2f} m/s")
        elif acceleration_phase == 'constant_at_v1':
            current_velocity = v1_real
            print(f"Frame {frame_idx}: Vận tốc hiện tại (ổn định ở v1_real) = {current_velocity:.2f} m/s")

        # ------------------------- Tính Số Khung Hình Cần Bỏ Qua -------------------------
        if v1_real > 0:
            skip_ratio = current_velocity / v1_real
            frames_to_skip = int(skip_ratio) - 1
            if frames_to_skip < 0:
                frames_to_skip = 0  # Tránh giá trị âm
        else:
            frames_to_skip = 0  # Tránh chia cho 0

        # Cập nhật frame_idx để bỏ qua các khung hình đã xử lý
        frame_idx += frames_to_skip + 1  # +1 để chuyển đến khung hình tiếp theo

    print("Hoàn thành xử lý dán đối tượng với gia tốc.")

def paste_object_with_sudden_velocity_change(nth_person, tracking_data, v1_real, v2_real):
    # ------------------------- Cấu Hình -------------------------
    inpainted_frames_folder = "./data_ext/gendata/01_background"  # Thư mục chứa khung hình đã xóa đối tượng
    original_frames_folder = "./data_ext/01_Train"  # Thư mục chứa khung hình gốc
    gendata_folder = "./data_ext/gendata/01_sudden_velocity_change"  # Thư mục lưu khung hình đầu ra cuối cùng

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

        # Đọc khung hình đã xóa đối tượng
        inpainted_frame_path = inpainted_frames_list[frame_index]
        inpainted_frame = cv2.imread(inpainted_frame_path)
        if inpainted_frame is None:
            print(f"Không thể đọc khung hình đã xóa đối tượng: {inpainted_frame_path}")
            frame_idx += 1
            continue

        # Đọc khung hình gốc
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

# ------------------------- Luồng Chính -------------------------
if __name__ == "__main__":
    nth_person = 1  # Chọn người thứ 1

    # Số thứ nhất là vận tốc và số thứ 2 là gia tốc
    v2_a_mapping = {
        3 : 1.4, # Người trượt ván
        3.3: 2.3, # Người chạy bộ
        4: 3.1, # Người đi xe đạp
        5: 4.5  #  Xe máy đi chậm
    }

    # Chọn ngẫu nhiên một v2_real từ danh sách
    v2_real = random.choice(list(v2_a_mapping.keys()))
    a_real = v2_a_mapping[v2_real]
    print(f"Vận tốc tối đa được chọn ngẫu nhiên: {v2_real} m/s")
    print(f"Gia tốc tương ứng: {a_real} m/s²")

    # Gọi hàm remove_background để xóa nền
    tracking_data, v1_real = remove_background(nth_person, v2_real, a_real)

    # Phương pháp 1: Dán đối tượng với gia tốc
    paste_object_with_acceleration(nth_person, tracking_data, v1_real, v2_real, a_real)

    # Phương pháp 2: Dán đối tượng với vận tốc thay đổi đột ngột
    paste_object_with_sudden_velocity_change(nth_person, tracking_data, v1_real, v2_real)
