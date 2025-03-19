import cv2
import os
import json
from typing import List, Dict

def load_tracking_data(tracking_data_path: str) -> List[Dict]:
    """
    Tải tracking_data từ tệp JSON.

    :param tracking_data_path: Đường dẫn tới tệp JSON chứa tracking data
    :return: Danh sách chứa thông tin tracking của đối tượng mục tiêu
    """
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    return tracking_data

def visualize_tracking_independent(nth_person: int, tracking_data: List[Dict], frames_folder: str, output_video_path: str = None):
    """
    Visualize the tracking of the nth_person by drawing bounding boxes and labels on frames.

    :param nth_person: Số thứ tự của người cần visualize (ví dụ: 1)
    :param tracking_data: Danh sách chứa thông tin tracking của đối tượng mục tiêu
                           Mỗi phần tử là một dictionary với keys 'frame_idx' và 'bbox'
    :param frames_folder: Thư mục chứa các khung hình gốc
    :param output_video_path: Đường dẫn lưu video kết quả. Nếu None, không lưu video
    """
    # Lấy danh sách các khung hình và sắp xếp theo thứ tự
    frames_list = sorted([
        os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder)
        if frame.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frames_list:
        print("Không tìm thấy khung hình trong thư mục:", frames_folder)
        return

    # Tạo một dictionary để truy cập tracking_data nhanh chóng theo frame_idx
    tracking_dict = {data['frame_idx']: data['bbox'] for data in tracking_data}

    # Lấy kích thước khung hình từ khung hình đầu tiên
    first_frame = cv2.imread(frames_list[0])
    if first_frame is None:
        print("Không thể đọc khung hình đầu tiên:", frames_list[0])
        return
    height, width, layers = first_frame.shape

    # Nếu output_video_path được cung cấp, khởi tạo VideoWriter
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho định dạng MP4
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
        if not out.isOpened():
            print("Không thể mở VideoWriter với đường dẫn:", output_video_path)
            out = None
    else:
        out = None

    # Tạo cửa sổ hiển thị (nếu không lưu video)
    if not out:
        cv2.namedWindow('Tracking Visualization', cv2.WINDOW_NORMAL)

    for idx, frame_path in enumerate(frames_list):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Không thể đọc khung hình: {frame_path}")
            continue

        if idx in tracking_dict:
            bbox = tracking_dict[idx]
            x1, y1, x2, y2 = bbox

            # Vẽ hộp bao
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ nhãn của đối tượng (nth_person)
            label = f'Person {nth_person}'
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Ghi khung hình vào video nếu có
        if out:
            out.write(frame)
        else:
            # Hiển thị khung hình trên cửa sổ
            cv2.imshow('Tracking Visualization', frame)
            # Nhấn 'q' để thoát sớm
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Giải phóng VideoWriter và đóng các cửa sổ hiển thị
    if out:
        out.release()
        print(f"Đã lưu video visualization vào: {output_video_path}")
    else:
        cv2.destroyAllWindows()
        print("Đã hoàn thành visualization trên cửa sổ hiển thị.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize tracking data for nth_person.")
    parser.add_argument('--nth_person', type=int, default=1, help='Số thứ tự của người cần visualize (ví dụ: 1)')
    parser.add_argument('--tracking_data_path', type=str, required=True, help='Đường dẫn tới tệp JSON chứa tracking data')
    parser.add_argument('--frames_folder', type=str, required=True, help='Thư mục chứa các khung hình gốc')
    parser.add_argument('--output_video_path', type=str, default=None, help='Đường dẫn lưu video kết quả. Nếu không cung cấp, hiển thị trực tiếp trên màn hình')

    args = parser.parse_args()

    # Tải tracking_data
    tracking_data = load_tracking_data(args.tracking_data_path)

    # Gọi hàm visualize
    visualize_tracking_independent(args.nth_person, tracking_data, args.frames_folder, args.output_video_path)
