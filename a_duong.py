import cv2
import torch
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import math

def track_nth_person(nth_person):
    # ------------------------- Cấu Hình Thư Mục -------------------------
    frames_folder = "/data_ext/01_Train"
    background_folder = "./data_ext/gendata/01_background"
    gendata_folder = "./data_ext/gendata/01"
    conf_threshold = 0.5
    tracking_class = 0  # Nếu muốn theo dõi tất cả các lớp, đặt thành None

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(gendata_folder, exist_ok=True)

    # ------------------------- Khởi Tạo Mô Hình -------------------------
    # Khởi tạo DeepSort
    tracker = DeepSort(max_age=30)

    # Khởi tạo YOLOv9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights="weights/yolov9-e.pt", device=device, fuse=True)
    model = AutoShape(model)

    # Tải tên các lớp từ file classes.names
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # Lấy danh sách các khung hình
    frames_list = sorted([
        os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])
    background_frames_list = sorted([
        os.path.join(background_folder, frame) for frame in os.listdir(background_folder)
        if frame.endswith('.jpg') or frame.endswith('.png')
    ])

    # ------------------------- Hiệu Chuẩn Tỷ Lệ -------------------------
    FPS = 24  # Số khung hình trên giây của video

    # Định nghĩa các tham số vận tốc và gia tốc thực tế
    v1_real = 1.39  # m/s (vận tốc đi bộ)
    v2_real = 4.17  # m/s (vận tốc đi xe đạp)
    a_real = 0.5   # m/s² (gia tốc)

    print(f"Vận tốc đi bộ: {v1_real:.2f} m/s")
    print(f"Vận tốc xe đạp: {v2_real:.2f} m/s")
    print(f"Gia tốc: {a_real:.2f} m/s²")

    # ------------------------- Hàm Thêm Nhiễu Gaussian -------------------------
    def add_gaussian_noise(image, mean=0, var=0.01):
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
        noisy = cv2.add(image, gauss)
        return noisy

    # ------------------------- Hàm Tính Vận Tốc Hiện Tại -------------------------
    def calculate_current_velocity(current_frame, v1_real, a_real, FPS, v2_real):
        time_elapsed = current_frame / FPS  # Thời gian đã trôi qua (giây)
        current_v = v1_real + a_real * time_elapsed
        if current_v > v2_real:
            current_v = v2_real
        return current_v

    # ------------------------- Khởi Tạo Biến Để Lưu Trữ -------------------------
    extracted_boxes = []
    target_track_id = None

    # ------------------------- Xử Lý Các Khung Hình -------------------------
    frame_idx = 0  # Sử dụng biến riêng để quản lý chỉ số khung hình

    while frame_idx < len(frames_list):
        frame_path = frames_list[frame_idx]
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Không thể đọc khung hình: {frame_path}")
            frame_idx += 1
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

        # Xác định target_track_id lần đầu tiên
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
            found_target = False
            for track in tracks:
                if track.is_confirmed() and track.track_id == target_track_id:
                    found_target = True
                    # Tính vận tốc hiện tại
                    current_velocity = calculate_current_velocity(frame_idx, v1_real, a_real, FPS, v2_real)
                    print(f"Frame {frame_idx}: Vận tốc hiện tại = {current_velocity:.2f} m/s")

                    # Tính số khung hình cần bỏ qua
                    if v1_real > 0:
                        skip_ratio = current_velocity / v1_real
                        frames_to_skip = int(skip_ratio) - 1
                        if frames_to_skip < 0:
                            frames_to_skip = 0
                    else:
                        frames_to_skip = 0  # Tránh chia cho 0

                    # Xác định chỉ số khung hình mục tiêu
                    target_frame_index = frame_idx + frames_to_skip + 1  # +1 để di chuyển tới khung hình tiếp theo

                    if target_frame_index >= len(frames_list):
                        print(f"Không thể bỏ qua {frames_to_skip} khung hình từ frame {frame_idx}. Đã vượt quá số lượng khung hình.")
                        frame_idx += 1
                        break

                    # Đọc khung hình mục tiêu
                    target_frame_path = frames_list[target_frame_index]
                    target_frame = cv2.imread(target_frame_path)
                    if target_frame is None:
                        print(f"Không thể đọc khung hình mục tiêu: {target_frame_path}")
                        frame_idx += 1
                        continue

                    # Phát hiện đối tượng trong khung hình mục tiêu
                    target_results = model(target_frame)
                    target_detect = []
                    for target_detect_object in target_results.pred[0]:
                        if len(target_detect_object) < 6:
                            continue  # Bỏ qua nếu không đủ thông tin
                        t_x1, t_y1, t_x2, t_y2, t_confidence, t_class_id = target_detect_object[:6]
                        t_x1, t_y1, t_x2, t_y2 = map(int, [t_x1, t_y1, t_x2, t_y2])
                        t_class_id = int(t_class_id)

                        if tracking_class is None:
                            if t_confidence < conf_threshold:
                                continue
                        else:
                            if t_class_id != tracking_class or t_confidence < conf_threshold:
                                continue

                        target_detect.append([[t_x1, t_y1, t_x2 - t_x1, t_y2 - t_y1], t_confidence, t_class_id])

                    # Cập nhật các tracks với DeepSort cho khung hình mục tiêu
                    target_tracks = tracker.update_tracks(target_detect, frame=target_frame)

                    # Tìm hộp bounding box của đối tượng mục tiêu trong khung hình mục tiêu
                    target_box = None
                    for t_track in target_tracks:
                        if t_track.is_confirmed() and t_track.track_id == target_track_id:
                            ltrb = t_track.to_ltrb()
                            tx1, ty1, tx2, ty2 = map(int, ltrb)
                            target_box = (tx1, ty1, tx2, ty2)
                            break

                    if target_box is not None:
                        tx1, ty1, tx2, ty2 = target_box
                        extracted_box = target_frame[ty1:ty2, tx1:tx2]
                        if extracted_box.size == 0:
                            print(f"Hộp trích xuất rỗng từ khung hình {target_frame_index}.")
                            continue
                        # Thêm nhiễu Gaussian vào hộp trích xuất
                        noisy_box = add_gaussian_noise(extracted_box)
                        extracted_boxes.append((noisy_box, (tx1, ty1, tx2, ty2)))
                        print(f"Đã trích xuất và thêm nhiễu vào hộp từ khung hình {target_frame_index}")

                        # Dán hộp vào khung hình nền
                        if len(extracted_boxes) > len(background_frames_list):
                            print("Đã đạt đến số lượng khung hình nền. Dừng quá trình dán.")
                            break

                        background_frame_path = background_frames_list[len(extracted_boxes) - 1]
                        background_frame = cv2.imread(background_frame_path)
                        if background_frame is None:
                            print(f"Không thể đọc khung hình nền: {background_frame_path}")
                            continue

                        # Resize hộp nhiễu để phù hợp với kích thước ban đầu
                        box_width = tx2 - tx1
                        box_height = ty2 - ty1
                        resized_box = cv2.resize(noisy_box, (box_width, box_height))

                        # Đảm bảo rằng tọa độ không vượt quá kích thước khung hình nền
                        y2_paste = min(ty2, background_frame.shape[0])
                        x2_paste = min(tx2, background_frame.shape[1])
                        y1_paste = max(ty1, 0)
                        x1_paste = max(tx1, 0)
                        resized_box = resized_box[0:(y2_paste - y1_paste), 0:(x2_paste - x1_paste)]

                        # Dán hộp vào khung hình nền
                        background_frame[y1_paste:y2_paste, x1_paste:x2_paste] = resized_box

                        # Lưu khung hình đã chỉnh sửa vào thư mục đầu ra
                        output_frame_path = os.path.join(gendata_folder, os.path.basename(background_frame_path))
                        cv2.imwrite(output_frame_path, background_frame)
                        print(f"Đã lưu khung hình đã chỉnh sửa vào: {output_frame_path}")

                        # Cập nhật frame_idx để bỏ qua các khung hình đã xử lý
                        frame_idx = target_frame_index
                    else:
                        print(f"Không tìm thấy đối tượng mục tiêu trong khung hình {target_frame_index}.")
                        frame_idx += 1
                    break  # Thoát khỏi vòng lặp for tracks
            if not found_target:
                # Nếu không tìm thấy đối tượng mục tiêu trong các tracks
                frame_idx += 1
        else:
            # Nếu chưa thiết lập target_track_id
            frame_idx += 1

    # ------------------------- Kết Thúc Xử Lý -------------------------
    cv2.destroyAllWindows()
    print("Hoàn thành xử lý tất cả các khung hình.")

# Chạy hàm với tham số chỉ định người thứ mấy sẽ được theo dõi (ví dụ: người thứ 3)
track_nth_person(3)
