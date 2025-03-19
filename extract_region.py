# import cv2

# # Load the image
# image_path = "/home/thieugt95/code/yolov9/data_ext/01_Train/002.jpg"
# image = cv2.imread(image_path)

# # Coordinates to paste into (41, 179, 55, 220)
# paste_x1, paste_y1, paste_x2, paste_y2 = 41, 179, 55, 220

# # Calculate the width and height of the paste region
# region_width = paste_x2 - paste_x1
# region_height = paste_y2 - paste_y1

# # Define the region to extract from the right side of the image (with 10 pixels padding)
# padding = 45
# extract_x1 = image.shape[1] - region_width - padding
# extract_y1 = paste_y1
# extract_x2 = image.shape[1] - padding
# extract_y2 = paste_y2

# # Extract the region from the right side of the image
# extracted_region = image[extract_y1:extract_y2, extract_x1:extract_x2]

# # Ensure the extracted region has the same size as the paste region
# if extracted_region.shape[1] != region_width or extracted_region.shape[0] != region_height:
#     raise ValueError("The extracted region size does not match the paste region size.")

# # Paste the extracted region into the specified coordinates
# image[paste_y1:paste_y2, paste_x1:paste_x2] = extracted_region
# print(image)
# # Save the modified image
# output_path = "/home/thieugt95/code/yolov9/data_ext/modified_image.jpg"
# cv2.imwrite(output_path, image)

# print(f"Modified image saved to {output_path}")
import cv2

# Load the image
image_path = "/home/thieugt95/code/yolov9/data_ext/01_Train/002.jpg"
image = cv2.imread(image_path)

# New coordinates to extract from (61, 185, 105, 220)
extract_x1, extract_y1, extract_x2, extract_y2 = 60, 180, 110, 260

# Extract the region from the image based on the new coordinates
extracted_region = image[extract_y1:extract_y2, extract_x1:extract_x2]

# Save the extracted region
output_path = "/home/thieugt95/code/yolov9/data_ext/extracted_region1.jpg"
cv2.imwrite(output_path, extracted_region)

print(f"Extracted region saved to {output_path}")


