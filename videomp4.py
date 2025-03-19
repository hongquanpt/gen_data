import cv2
import os



def images_to_video(input_folder, output_file, fps=30):
    # Lấy danh sách các ảnh trong thư mục
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Sắp xếp ảnh theo thứ tự (tùy chọn)

    # Đọc ảnh đầu tiên để lấy kích thước khung hình
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # Khởi tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Thêm từng ảnh vào video
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Giải phóng bộ nhớ
    video.release()
    print(f"Video created successfully: {output_file}")

# Thư mục chứa ảnh và đường dẫn file video đầu ra
input_folder = "./data_ext/gendata/01_acceleration"  # Thay đường dẫn thư mục ảnh
output_file = "./data_ext/gendata/01_acceleration/output_video.mp4"        # Đường dẫn lưu video đầu ra
fps = 24                                # Số khung hình trên giây

# Gọi hàm tạo video
images_to_video(input_folder, output_file, fps)
