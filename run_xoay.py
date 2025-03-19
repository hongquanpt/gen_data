import os
import subprocess

# Thư mục gốc
base_folder = "C:/Users/Admin/Desktop/VAD/dataset/ped2/training"

# Lặp qua tất cả các thư mục con
for subdir in os.listdir(base_folder):
    subdir_path = os.path.join(base_folder, subdir)
    
    # Kiểm tra nếu là thư mục
    if os.path.isdir(subdir_path):
        # Tạo lệnh để chạy
        command = f"python xoay1.py --input_folder {subdir_path}"
        
        # In ra lệnh để theo dõi
        print(f"Running command: {command}")
        
        # Thực thi lệnh
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command for {subdir_path}: {e}")
